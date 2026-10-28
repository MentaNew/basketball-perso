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


@pytest.mark.parametrize("remarks,season,day,expected", [
    ("at London, England", 2019, date(2019, 1, 17), True),
    ("at Mexico City, Mexico", 2024, date(2023, 11, 9), True),
    ("at Hartford, CT", 1990, date(1990, 3, 9), False),        # Celtics' second home
    ("at Toronto, Canada (Maple Leaf Gardens)", 1998, date(1998, 4, 1), False),
    ("at Kingdome", 1989, date(1989, 5, 14), False),
    (None, 2020, date(2020, 8, 1), True),                       # Orlando bubble
    (None, 2020, date(2020, 3, 1), False),
    ("NBA Cup", 2025, date(2024, 11, 12), False),
])
def test_neutral_site_rule(remarks, season, day, expected):
    assert parsers.is_neutral_site(remarks, season, day) is expected


def test_schedule_has_neutral_flag():
    df = parsers.parse_schedule(page("NBA_2024_games-october"), 2024)
    assert df["neutral"].dtype == bool and not df["neutral"].any()


def test_last_completed_season():
    from etl.pipeline import last_completed_season
    assert last_completed_season(date(2026, 10, 3)) == 2026
    assert last_completed_season(date(2026, 3, 1)) == 2025


def test_win_totals_full_season():
    df = parsers.parse_win_totals(page("win_totals_2025"))
    assert len(df) == 30
    bos = df.set_index("team_name").loc["Boston Celtics"]
    assert (bos.line, bos.over_odds, bos.under_odds, bos.actual_wins) == (58.5, -105, -115, 61)


def test_win_totals_partial_archive_keeps_only_real_lines():
    df = parsers.parse_win_totals(page("win_totals_1995"))   # 1995: only 9 of 27 teams have a line
    assert len(df) == 9 and df["line"].notna().all()


def test_win_totals_page_without_table_is_empty():
    assert parsers.parse_win_totals("<html><p>no lines yet</p></html>").empty


def test_salaries_from_hidden_table():
    df = parsers.parse_salaries(page("BOS_1994_team"))       # table sits inside an HTML comment
    assert len(df) == 15 and df["bref_id"].is_unique
    parish = df.set_index("bref_id").loc["parisro01"]
    assert (parish["name"], parish["salary"]) == ("Robert Parish", 4_000_000)


def test_salary_caps():
    caps = parsers.parse_salary_caps(page("salary_cap_history"))
    assert caps[1994] == 15_175_000 and caps[2026] == 154_647_000
    assert min(caps) == 1985


def test_salaries_sum_a_players_multiple_contracts():
    df = parsers.parse_salaries(page("HOU_2015_team"))      # Luis Scola is listed twice
    assert df["bref_id"].is_unique
    assert df.set_index("bref_id").loc["scolalu01", "salary"] == 6_541_000 + 6_172_538


def test_roster_ids_come_from_player_links():
    ids = parsers.parse_roster(page("OKC_2027_team"))
    assert len(ids) == 17 and "gilgesh01" in ids and len(set(ids)) == 17
