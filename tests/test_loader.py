"""Loader integration tests against a throwaway PostgreSQL database.

Set TEST_DATABASE_URL (see .env.example).  The schema there is dropped and recreated for
every test module, so it must never point at the real database.  Skipped when unset.
"""
import gzip
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()
TEST_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_URL, reason="TEST_DATABASE_URL not set")

FIXTURES = Path(__file__).parent / "fixtures"
SEASON = 2025


def page(name: str) -> str:
    with gzip.open(FIXTURES / f"{name}.html.gz", "rt", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def db():
    """Point db.connection at the test database with a fresh schema."""
    from db import connection

    if TEST_URL == os.environ.get("DATABASE_URL"):
        pytest.fail("TEST_DATABASE_URL must differ from DATABASE_URL (the test wipes it)")
    real = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_URL
    connection.get_engine.cache_clear()
    from db.init_db import init_db

    engine = connection.get_engine()
    init_db(engine, drop=True)
    yield connection
    engine.dispose()
    connection.get_engine.cache_clear()
    if real is not None:
        os.environ["DATABASE_URL"] = real


def count(db, sql: str) -> int:
    return int(db.query_df(sql).iat[0, 0])


@pytest.fixture(scope="module")
def parsed():
    from etl import parsers
    return {
        "players": parsers.merge_player_tables(
            parsers.parse_player_table(page("NBA_2025_per_game"), "per_game"),
            parsers.parse_player_table(page("NBA_2025_advanced"), "advanced")),
        "standings": parsers.parse_standings(page("NBA_2025_standings")),
        # an October schedule page relabelled as 2025: the same 30 teams, so quotas apply
        "games": parsers.parse_schedule(page("NBA_2024_games-october"), SEASON),
    }


def test_games_need_standings_first(db, parsed):
    from etl import loader
    with pytest.raises(RuntimeError, match="standings"):
        loader.load_games(parsed["games"], SEASON)


def test_player_load_is_idempotent_and_wide(db, parsed):
    from etl import loader
    for _ in range(2):
        loader.load_players(parsed["players"], SEASON)
    n_rows = count(db, "SELECT COUNT(*) FROM player_seasons")
    assert n_rows == len(parsed["players"])                       # no duplicates
    assert count(db, "SELECT COUNT(*) FROM players") == parsed["players"].bref_id.nunique()
    # per-game and advanced columns live in the same wide row
    assert count(db, "SELECT COUNT(*) FROM player_seasons WHERE pts IS NULL OR per IS NULL") == 0
    luka = db.query_df("""SELECT s.team_abr, s.is_primary, s.team_list FROM player_seasons s
                          JOIN players p ON p.id = s.player_id WHERE p.bref_id = 'doncilu01'
                          ORDER BY s.team_abr""")
    assert luka.to_dict("list") == {"team_abr": ["DAL", "LAL", "TOT"],
                                    "is_primary": [False, False, True],
                                    "team_list": ["DAL", "LAL", "DAL,LAL"]}


def test_player_name_corrections_propagate(db, parsed):
    from etl import loader
    renamed = parsed["players"].copy()
    renamed.loc[renamed.bref_id == "gilgesh01", "name"] = "Shai Gilgeous Alexander"
    loader.load_players(renamed, SEASON)
    assert db.query_df("SELECT name, search_name FROM players WHERE bref_id = 'gilgesh01'"
                       ).iloc[0].tolist() == ["Shai Gilgeous Alexander", "shai gilgeous alexander"]
    loader.load_players(parsed["players"], SEASON)


def test_reload_removes_rows_that_disappeared(db, parsed):
    from etl import loader
    fewer = parsed["players"][parsed["players"].bref_id != "gilgesh01"]
    loader.load_players(fewer, SEASON)
    assert count(db, """SELECT COUNT(*) FROM player_seasons s JOIN players p ON p.id = s.player_id
                        WHERE p.bref_id = 'gilgesh01'""") == 0
    loader.load_players(parsed["players"], SEASON)
    assert count(db, "SELECT COUNT(*) FROM player_seasons") == len(parsed["players"])


def test_standings_and_games_idempotent(db, parsed):
    from etl import loader
    for _ in range(2):
        loader.load_standings(parsed["standings"], SEASON)
        loader.load_games(parsed["games"], SEASON)
    assert count(db, "SELECT COUNT(*) FROM team_seasons") == 30
    assert count(db, "SELECT COUNT(*) FROM games") == len(parsed["games"])
    assert count(db, "SELECT COUNT(*) FROM games WHERE game_type <> 'regular'") == 0
    g = db.query_df("SELECT * FROM games WHERE game_id = '202310240DEN'").iloc[0]
    assert (g.away_team, g.away_score, g.home_team, g.home_score, g.home_win, g.neutral) == (
        "LAL", 107, "DEN", 119, True, False)
    # dropping a game from the source removes it from the DB
    loader.load_games(parsed["games"][parsed["games"].game_id != "202310240DEN"], SEASON)
    assert count(db, "SELECT COUNT(*) FROM games WHERE game_id = '202310240DEN'") == 0


def test_franchise_views(db, parsed):
    from etl import loader
    loader.load_standings(parsed["standings"], SEASON)
    fr = db.query_df("SELECT * FROM franchises WHERE franchise_id = 'OKC'").iloc[0]
    assert (fr.current_abr, fr.full_name) == ("OKC", "Oklahoma City Thunder")
    assert count(db, "SELECT COUNT(*) FROM teams") == 30
