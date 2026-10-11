"""Write a season's parsed DataFrames into PostgreSQL.

Pattern: "replace the season". Inside one transaction we delete the season's rows and
insert the fresh ones. That makes every load idempotent (running it twice gives the same
result) and self-cleaning (rows that vanished from the source disappear), and if anything
fails halfway the transaction rolls back and the old data stays intact.

Players are the exception: a player spans many seasons, so they are upserted by bref_id.
"""
import pandas as pd
from sqlalchemy import Connection, delete, func, select
from sqlalchemy.dialects.postgresql import insert

from db.connection import get_engine
from db.models import EtlCheckpoint, Game, Player, PlayerSalary, PlayerSeason, SalaryCap, TeamSeason, WinTotal
from etl.parsers import search_name
from etl.teams import franchise_of


def load_players(df: pd.DataFrame, season: int) -> int:
    """Replace the season's player_seasons rows (df comes from scraper.get_players)."""
    with get_engine().begin() as conn:
        player_ids = upsert_players(conn, df, season)
        rows = df.assign(season=season, player_id=df["bref_id"].map(player_ids))
        rows["age"] = rows["age"].astype("Int64")
        columns = [c.name for c in PlayerSeason.__table__.columns if c.name != "id"]
        conn.execute(delete(PlayerSeason).where(PlayerSeason.season == season))
        rows[columns].to_sql(PlayerSeason.__tablename__, conn, if_exists="append", index=False)
    return len(rows)


def upsert_players(conn: Connection, df: pd.DataFrame, season: int) -> dict[str, int]:
    """Insert new players, update known ones; return {bref_id: players.id}."""
    people = df.drop_duplicates("bref_id")
    records = [
        {
            "bref_id": p.bref_id,
            "name": p.name,
            "search_name": search_name(p.name),
            "birth_year": None if pd.isna(p.age) else int(season - p.age - 1),
        }
        for p in people.itertuples()
    ]
    stmt = insert(Player).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["bref_id"],
        set_={
            "name": stmt.excluded.name,  # BRef sometimes corrects spellings
            "search_name": stmt.excluded.search_name,
            # age is "as of Feb 1", so season - age - 1 can be one year late; keep the
            # earliest estimate (Postgres LEAST ignores NULLs)
            "birth_year": func.least(Player.birth_year, stmt.excluded.birth_year),
        },
    )
    conn.execute(stmt)
    found = conn.execute(select(Player.bref_id, Player.id)
                         .where(Player.bref_id.in_([r["bref_id"] for r in records])))
    return dict(found.all())


def load_standings(df: pd.DataFrame, season: int) -> int:
    rows = df.assign(season=season, franchise_id=df["team_abr"].map(franchise_of))
    with get_engine().begin() as conn:
        conn.execute(delete(TeamSeason).where(TeamSeason.season == season))
        rows.to_sql(TeamSeason.__tablename__, conn, if_exists="append", index=False)
    return len(rows)


def load_games(df: pd.DataFrame, season: int) -> int:
    """Replace the season's games, tagging each with its game_type.

    The game types need each team's W+L from the standings, so load standings first.
    """
    with get_engine().begin() as conn:
        standings = conn.execute(select(TeamSeason.team_abr, TeamSeason.w + TeamSeason.l)
                                 .where(TeamSeason.season == season)).all()
        if not standings:
            raise RuntimeError(f"no standings for {season}: run the standings step first")
        games = df.sort_values(["game_date", "game_id"], ignore_index=True)
        games["game_type"] = infer_game_types(games, quota=dict(standings))
        conn.execute(delete(Game).where(Game.season == season))
        games.to_sql(Game.__tablename__, conn, if_exists="append", index=False)
    return len(games)


def infer_game_types(games: pd.DataFrame, quota: dict[str, int]) -> list[str]:
    """'regular' | 'play_in' | 'playoff' | 'cup_final' for date-ordered games.

    BRef's schedule doesn't say which games are playoffs, so we infer it:
    - "Play-In Game" in remarks -> play_in.
    - The season's last "NBA Cup" / "In-Season Tournament" game is the Cup final; it doesn't
      count in the standings (2024 onward) -> cup_final.
    - Otherwise a game is regular while both teams are still under their W+L total (`quota`)
      from the standings. Everything after that is a playoff game.
    """
    remarks = games["remarks"].fillna("").str.lower().tolist()
    cup_games = [i for i, r in enumerate(remarks) if r in ("nba cup", "in-season tournament")]
    cup_final = cup_games[-1] if cup_games else None
    played: dict[str, int] = {}
    types = []
    for i, (home, away) in enumerate(zip(games["home_team"], games["away_team"], strict=True)):
        if remarks[i].startswith("play-in"):
            types.append("play_in")
        elif i == cup_final:
            types.append("cup_final")
        elif played.get(home, 0) < quota.get(home, 0) and played.get(away, 0) < quota.get(away, 0):
            types.append("regular")
            played[home] = played.get(home, 0) + 1
            played[away] = played.get(away, 0) + 1
        else:
            types.append("playoff")
    return types


# ---------------------------------------------------------------- money: betting lines, payroll, cap

# Odds-site team names that differ from BRef's name for the same season.
TEAM_NAME_ALIASES = {"New Orleans Hornets": "New Orleans/Oklahoma City Hornets"}  # 2006-07 (Katrina)


def load_win_totals(df: pd.DataFrame, season: int) -> int:
    """Replace the season's win totals, mapping team names to abbreviations via the standings."""
    with get_engine().begin() as conn:
        names = dict(conn.execute(select(TeamSeason.full_name, TeamSeason.team_abr)
                                  .where(TeamSeason.season == season)).all())
        team_abr = df["team_name"].map(lambda n: names.get(n) or names.get(TEAM_NAME_ALIASES.get(n)))
        if team_abr.isna().any():
            unknown = df.loc[team_abr.isna(), "team_name"].tolist()
            raise ValueError(f"unknown team names in {season}: {unknown}")
        rows = df.drop(columns="team_name").assign(season=season, team_abr=team_abr)
        conn.execute(delete(WinTotal).where(WinTotal.season == season))
        rows.to_sql(WinTotal.__tablename__, conn, if_exists="append", index=False)
    return len(rows)


def load_salaries(data: tuple[pd.DataFrame, int | None], season: int) -> int:
    """Replace the season's player salaries and salary cap (data comes from scraper.get_salaries)."""
    salaries, cap = data
    with get_engine().begin() as conn:
        conn.execute(delete(PlayerSalary).where(PlayerSalary.season == season))
        conn.execute(delete(SalaryCap).where(SalaryCap.season == season))
        rows = salaries.assign(season=season)
        rows.to_sql(PlayerSalary.__tablename__, conn, if_exists="append", index=False)
        if cap:
            conn.execute(insert(SalaryCap).values(season=season, cap=cap))
    return len(salaries)


# ---------------------------------------------------------------- checkpoint

def completed_steps() -> set[tuple[int, str]]:
    with get_engine().connect() as conn:
        rows = conn.execute(select(EtlCheckpoint.season, EtlCheckpoint.step))
        return {(season, step) for season, step in rows}


def mark_done(season: int, step: str, rows: int) -> None:
    stmt = insert(EtlCheckpoint).values(season=season, step=step, rows=rows)
    stmt = stmt.on_conflict_do_update(index_elements=["season", "step"],
                                      set_={"rows": rows, "completed_at": func.now()})
    with get_engine().begin() as conn:
        conn.execute(stmt)
