"""Parser tests on saved Basketball-Reference pages (tests/fixtures/*.html.gz)."""
import gzip
from datetime import date
from pathlib import Path

import pytest

from etl import parsers

FIXTURES = Path(__file__).parent / "fixtures"


def page(name: str) -> str:
    with gzip.open(FIXTURES / f"{name}.html.gz", "rt", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def per_game_2025():
    return parsers.parse_player_table(page("NBA_2025_per_game"), "per_game")


def test_per_game_has_bref_ids_and_real_teams(per_game_2025):
    df = per_game_2025
    assert df["bref_id"].notna().all()
    assert not (df["team_abr"] == "UNK").any()          # §7.4
    sga = df[df.bref_id == "gilgesh01"].iloc[0]
    assert (sga["name"], sga["team_abr"], sga["pts"], sga["g"]) == (
        "Shai Gilgeous-Alexander", "OKC", 32.7, 76)


def test_traded_player_gets_tot_primary_row(per_game_2025):
    luka = per_game_2025[per_game_2025.bref_id == "doncilu01"].set_index("team_abr")
    assert set(luka.index) == {"TOT", "DAL", "LAL"}     # "2TM" normalised to TOT
    assert luka.loc["TOT", "is_primary"]
    assert not luka.loc["DAL", "is_primary"]
    assert luka.loc["TOT", "team_list"] == "DAL,LAL"


def test_one_primary_row_per_player(per_game_2025):
    primary = per_game_2025[per_game_2025.is_primary]
    assert primary["bref_id"].is_unique
    assert set(primary["bref_id"]) == set(per_game_2025["bref_id"])


def test_old_era_per_game_parses():
    df = parsers.parse_player_table(page("NBA_1985_per_game"), "per_game")
    assert len(df) > 300 and not (df.team_abr == "UNK").any()
    bird = df[df.bref_id == "birdla01"].iloc[0]
    assert bird["name"] == "Larry Bird" and bird["team_abr"] == "BOS"


def test_advanced_table_id_and_columns():
    df = parsers.parse_player_table(page("NBA_2025_advanced"), "advanced")
    row = df[df.bref_id == "bridgmi01"].iloc[0]
    assert row["per"] == 14.0 and row["ts_pct"] == 0.585 and row["mp_total"] == 3036


@pytest.mark.parametrize("name,n_teams", [("NBA_1990_standings", 27), ("NBA_2025_standings", 30)])
def test_standings_both_conferences(name, n_teams):
    df = parsers.parse_standings(page(name))      # 1990 only has division tables (§7.3)
    assert len(df) == n_teams
    assert set(df.conference) == {"E", "W"}
    assert df["team_abr"].str.fullmatch(r"[A-Z0-9]{3}").all()
    assert df["division"].notna().all()


def test_standings_values_2025():
    df = parsers.parse_standings(page("NBA_2025_standings")).set_index("team_abr")
    okc = df.loc["OKC"]
    assert (okc.w, okc.l, okc.conference, okc.made_playoffs) == (68, 14, "W", True)
    assert okc.full_name == "Oklahoma City Thunder"   # '*' stripped
    assert okc.gb == 0.0


def test_schedule_home_away_not_inverted():
    df = parsers.parse_schedule(page("NBA_2024_games-october"), 2024)
    g = df.set_index("game_id").loc["202310240DEN"]   # LAL @ DEN, Denver won 119-107 (§7.1)
    assert (g.away_team, g.home_team, g.away_score, g.home_score) == ("LAL", "DEN", 107, 119)
    assert g.home_win
    assert g.game_date == date(2023, 10, 24)


def test_schedule_marks_play_in():
    df = parsers.parse_schedule(page("NBA_2024_games-april"), 2024)
    assert (df["remarks"] == "Play-In Game").sum() == 6
    assert df["game_id"].is_unique


def test_schedule_month_discovery_handles_bubble():
    months = parsers.parse_schedule_months(page("NBA_2020_games"), 2020)
    assert months[0] == "october-2019" and months[-1] == "october-2020"
    assert {"july", "august", "september"} <= set(months)
    assert "april" not in months


def test_missing_table_raises():
    with pytest.raises(parsers.TableNotFound):
        parsers.find_table(page("NBA_2024_games-october"), "per_game_stats")


def test_search_name_normalises_accents():
    assert parsers.search_name("Nikola Jokić") == "nikola jokic"
    assert parsers.clean_name("Michael Jordan*") == "Michael Jordan"
