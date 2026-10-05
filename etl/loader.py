"""Idempotent loads of parsed DataFrames into PostgreSQL.

Writes are `INSERT ... ON CONFLICT DO UPDATE`, and each load also deletes that season's rows
that are no longer in the source, so re-running a season makes the DB match the pages
exactly (no stale rows, no duplicates).
"""
import numpy as np
import pandas as pd
from sqlalchemy import delete, func, select, tuple_
from sqlalchemy.dialects.postgresql import insert

from db.connection import get_engine
from db.models import (
    ADVANCED_COLS,
    PER_GAME_COLS,
    EtlCheckpoint,
    Game,
    Player,
    PlayerSeason,
    TeamSeason,
)
from etl.parsers import search_name
from etl.teams import franchise_of


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of dicts with NaN/NA replaced by None and numpy scalars unwrapped."""
    out = []
    for rec in df.astype(object).to_dict("records"):
        out.append({k: (None if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA
                        else v.item() if isinstance(v, np.generic) else v)
                    for k, v in rec.items()})
    return out


def _upsert(conn, table, records: list[dict], key: list[str], update: list[str]) -> int:
    if not records:
        return 0
    stmt = insert(table)
    stmt = stmt.on_conflict_do_update(
        index_elements=key, set_={c: stmt.excluded[c] for c in update}
    )
    return conn.execute(stmt, records).rowcount


def upsert_players(conn, df: pd.DataFrame, season: int) -> dict[str, int]:
    """Upsert players by bref_id; return {bref_id: player_id}."""
    people = df.drop_duplicates("bref_id")[["bref_id", "name", "age"]].copy()
    people["search_name"] = people["name"].map(search_name)
    people["birth_year"] = (season - people["age"] - 1).astype("Int64")
    recs = _records(people.drop(columns="age"))
    stmt = insert(Player)
    stmt = stmt.on_conflict_do_update(
        index_elements=["bref_id"],
        set_={
            "name": stmt.excluded.name,
            "search_name": stmt.excluded.search_name,
            # keep the earliest estimate (the age-at-Feb-1 rule is off by one for some
            # birthdays); Postgres LEAST ignores NULLs
            "birth_year": func.least(Player.birth_year, stmt.excluded.birth_year),
        },
    )
    conn.execute(stmt, recs)
    ids = conn.execute(
        select(Player.bref_id, Player.id).where(Player.bref_id.in_(people["bref_id"].tolist()))
    ).all()
    return dict(ids)


def upsert_player_seasons(df: pd.DataFrame, season: int, kind: str) -> int:
    stat_cols = PER_GAME_COLS if kind == "per_game" else ADVANCED_COLS
    with get_engine().begin() as conn:
        id_map = upsert_players(conn, df, season)
        rows = df.assign(player_id=df["bref_id"].map(id_map), season=season)
        rows["age"] = rows["age"].astype("Int64")
        cols = ["player_id", "season", "team_abr", "team_list", "is_primary", "age", "pos", *stat_cols]
        update = ["team_list", "is_primary", "age", "pos", *stat_cols]
        n = _upsert(conn, PlayerSeason, _records(rows[cols]),
                    ["player_id", "season", "team_abr"], update)
        if kind == "per_game":  # per_game defines the season's row set; advanced only fills it
            keys = list(zip(rows["player_id"], rows["team_abr"], strict=True))
            conn.execute(delete(PlayerSeason).where(
                PlayerSeason.season == season,
                tuple_(PlayerSeason.player_id, PlayerSeason.team_abr).not_in(keys)))
        return n


def upsert_standings(df: pd.DataFrame, season: int) -> int:
    rows = df.assign(season=season, franchise_id=df["team_abr"].map(franchise_of))
    cols = [c.name for c in TeamSeason.__table__.columns if c.name != "id"]
    with get_engine().begin() as conn:
        n = _upsert(conn, TeamSeason, _records(rows[cols]), ["team_abr", "season"],
                    [c for c in cols if c not in ("team_abr", "season")])
        conn.execute(delete(TeamSeason).where(TeamSeason.season == season,
                                              TeamSeason.team_abr.not_in(rows["team_abr"].tolist())))
        return n


def upsert_games(df: pd.DataFrame, season: int) -> int:
    """Load a season's games and tag their game_type.  Needs that season's standings."""
    engine = get_engine()
    standings = pd.read_sql(
        select(TeamSeason.team_abr, TeamSeason.w, TeamSeason.l).where(TeamSeason.season == season),
        engine)
    if standings.empty:
        raise RuntimeError(f"no standings for {season}: run the standings step first")
    quota = dict(zip(standings.team_abr, (standings.w + standings.l).astype(int), strict=True))
    df = df.sort_values(["game_date", "game_id"], ignore_index=True)
    df["game_type"] = infer_game_types(df, quota)
    cols = [c.name for c in Game.__table__.columns]
    with engine.begin() as conn:
        n = _upsert(conn, Game, _records(df[cols]), ["game_id"], [c for c in cols if c != "game_id"])
        conn.execute(delete(Game).where(Game.season == season,
                                        Game.game_id.not_in(df["game_id"].tolist())))
    return n


def infer_game_types(games: pd.DataFrame, quota: dict[str, int]) -> list[str]:
    """regular / play_in / playoff / cup_final for date-ordered `games` (pure function).

    BRef's schedule doesn't mark playoffs, so a team's first W+L games (`quota`, from the
    standings) are its regular season; 'Play-In Game' remarks mark the play-in.  The NBA Cup
    final (the season's last 'NBA Cup' / 'In-Season Tournament' game, 2024+) doesn't count
    toward the standings.
    """
    remarks = games["remarks"].fillna("").str.lower().tolist()
    cup = [i for i, r in enumerate(remarks) if r in ("nba cup", "in-season tournament")]
    cup_final = cup[-1] if cup else None
    played: dict[str, int] = {}
    types = []
    for i, (home, away) in enumerate(zip(games["home_team"], games["away_team"], strict=True)):
        if remarks[i].startswith("play-in"):
            types.append("play_in")
        elif i == cup_final:
            types.append("cup_final")
        elif played.get(home, 0) < quota.get(home, 0) and played.get(away, 0) < quota.get(away, 0):
            types.append("regular")
            played[home], played[away] = played.get(home, 0) + 1, played.get(away, 0) + 1
        else:
            types.append("playoff")
    return types


# ---------------------------------------------------------------- checkpoint

def completed_steps() -> set[tuple[int, str]]:
    with get_engine().connect() as conn:
        return {tuple(r) for r in conn.execute(select(EtlCheckpoint.season, EtlCheckpoint.step))}


def mark_done(season: int, step: str, rows: int) -> None:
    stmt = insert(EtlCheckpoint).values(season=season, step=step, rows=rows)
    stmt = stmt.on_conflict_do_update(index_elements=["season", "step"],
                                      set_={"rows": rows, "completed_at": func.now()})
    with get_engine().begin() as conn:
        conn.execute(stmt)
