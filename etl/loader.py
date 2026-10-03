"""Idempotent upserts of parsed DataFrames into PostgreSQL.

All writes are `INSERT ... ON CONFLICT DO UPDATE`, so re-running a season refreshes it
instead of silently keeping stale rows.
"""
import numpy as np
import pandas as pd
from sqlalchemy import bindparam, func, select
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
            # keep the earliest estimate: the age-at-Feb-1 rule is off by one for some birthdays
            "birth_year": stmt.excluded.birth_year,
        },
        where=(Player.birth_year.is_(None)) | (stmt.excluded.birth_year < Player.birth_year),
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
        return _upsert(conn, PlayerSeason, _records(rows[cols]),
                       ["player_id", "season", "team_abr"], update)


def upsert_standings(df: pd.DataFrame, season: int) -> int:
    rows = df.assign(season=season, franchise_id=df["team_abr"].map(franchise_of))
    cols = [c.name for c in TeamSeason.__table__.columns if c.name != "id"]
    with get_engine().begin() as conn:
        return _upsert(conn, TeamSeason, _records(rows[cols]), ["team_abr", "season"],
                       [c for c in cols if c not in ("team_abr", "season")])


def upsert_games(df: pd.DataFrame, season: int) -> int:
    cols = [c.name for c in Game.__table__.columns if c.name != "game_type"]
    with get_engine().begin() as conn:
        n = _upsert(conn, Game, _records(df[cols]), ["game_id"],
                    [c for c in cols if c != "game_id"])
    classify_game_types(season)
    return n


def classify_game_types(season: int) -> None:
    """Tag games as regular / play_in / playoff / cup_final.

    BRef's schedule doesn't mark playoffs, so a team's first W+L games (from the
    standings) are its regular season; 'Play-In Game' remarks mark the play-in.  The NBA
    Cup final (the season's last 'NBA Cup' game, 2024+) doesn't count toward standings.
    """
    engine = get_engine()
    games = pd.read_sql(
        select(Game.game_id, Game.game_date, Game.home_team, Game.away_team, Game.remarks)
        .where(Game.season == season).order_by(Game.game_date, Game.game_id), engine)
    standings = pd.read_sql(
        select(TeamSeason.team_abr, TeamSeason.w, TeamSeason.l).where(TeamSeason.season == season),
        engine)
    if games.empty or standings.empty:
        return
    quota = dict(zip(standings.team_abr, standings.w + standings.l, strict=True))
    remarks = games["remarks"].fillna("").str.lower()
    cup_games = games.index[remarks.isin(["nba cup", "in-season tournament"])]
    cup_final = cup_games[-1] if len(cup_games) else None
    played: dict[str, int] = {}
    types = []
    for g in games.itertuples():
        if remarks[g.Index].startswith("play-in"):
            types.append("play_in")
            continue
        if g.Index == cup_final:
            types.append("cup_final")
            continue
        h, a = played.get(g.home_team, 0), played.get(g.away_team, 0)
        if h < quota.get(g.home_team, 0) and a < quota.get(g.away_team, 0):
            types.append("regular")
            played[g.home_team], played[g.away_team] = h + 1, a + 1
        else:
            types.append("playoff")
    table = Game.__table__
    stmt = (table.update().where(table.c.game_id == bindparam("gid"))
            .values(game_type=bindparam("gtype")))
    with engine.begin() as conn:
        conn.execute(stmt, [{"gid": gid, "gtype": t}
                            for gid, t in zip(games.game_id, types, strict=True)])


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
