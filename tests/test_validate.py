"""Mutation tests for the validation gate: re-inject past bugs and check they are caught.

A gate that always passes proves nothing until you've seen it fail. Each test corrupts the
loaded database inside a transaction, runs every check, and **always rolls back**, so the
data is never changed. Needs a fully loaded DATABASE_URL; skipped otherwise.
"""
import pytest
from sqlalchemy import text

from etl.pipeline import SEASONS, STEPS
from etl.validate import run_checks


@pytest.fixture(scope="module")
def engine():
    from db.connection import get_engine
    engine = get_engine()
    try:
        with engine.connect() as conn:
            done = conn.execute(text("SELECT COUNT(*) FROM etl_checkpoint")).scalar()
    except Exception:
        pytest.skip("DATABASE_URL is not reachable")
    if done < len(SEASONS) * len(STEPS):
        pytest.skip("DATABASE_URL is not fully loaded (run python -m etl.pipeline)")
    return engine


def failed_checks(engine, corruption: str) -> set[str]:
    with engine.connect() as conn:
        transaction = conn.begin()
        try:
            conn.execute(text(corruption))
            return {name for name, bad in run_checks(conn, SEASONS).items() if not bad.empty}
        finally:
            transaction.rollback()


def test_clean_database_passes(engine):
    assert failed_checks(engine, "SELECT 1") == set()


@pytest.mark.parametrize("corruption,expected_check", [
    # v1 bug §7.1: home and away scores swapped
    ("UPDATE games SET home_score = away_score, away_score = home_score, home_win = NOT home_win "
     "WHERE season = 2024", "standings W-L equals W-L recomputed from the games"),
    # v1 bug §7.3: West standings missing
    ("DELETE FROM team_seasons WHERE season = 2010 AND conference = 'W'",
     "22-30 teams per season, in both conferences"),
    # v1 bug §7.4: player teams lost
    ("UPDATE player_seasons SET team_abr = 'UNK' WHERE season = 2010 AND is_primary AND team_abr <> 'TOT'",
     "no player row has an unknown team"),
    ("DELETE FROM etl_checkpoint WHERE season = 1995 AND step = 'schedule'",
     "every ETL step finished for every season"),
    ("UPDATE games SET game_type = 'regular' WHERE game_type = 'cup_final'",
     "standings W-L equals W-L recomputed from the games"),
    ("UPDATE team_seasons SET franchise_id = team_abr WHERE team_abr = 'SEA'",
     "every retired abbreviation belongs to a current franchise (etl/teams.py)"),
])
def test_injected_bug_is_caught(engine, corruption, expected_check):
    assert expected_check in failed_checks(engine, corruption)
