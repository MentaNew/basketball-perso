"""Game-type inference (etl.loader.infer_game_types) on hand-built seasons."""
import pandas as pd

from etl.loader import infer_game_types


def season(rows):
    return pd.DataFrame(rows, columns=["home_team", "away_team", "remarks"])


def test_first_wl_games_are_regular_then_playoffs():
    # two-game regular season for everyone, then a playoff series
    games = season([("A", "B", None), ("C", "D", None), ("B", "A", None), ("D", "C", None),
                    ("A", "C", None), ("C", "A", None)])
    assert infer_game_types(games, {"A": 2, "B": 2, "C": 2, "D": 2}) == [
        "regular", "regular", "regular", "regular", "playoff", "playoff"]


def test_play_in_is_tagged_by_remarks_and_uses_no_quota():
    games = season([("A", "B", None), ("B", "A", "Play-In Game"), ("B", "A", None)])
    assert infer_game_types(games, {"A": 2, "B": 2}) == ["regular", "play_in", "regular"]


def test_cup_final_is_last_cup_game_and_not_counted():
    # A and B each play 2 counting games; the Cup final in between must not use up quota,
    # otherwise A's last regular-season game would be mislabelled a playoff game (2024-25 bug)
    games = season([("A", "B", "NBA Cup"), ("B", "A", "NBA Cup"), ("A", "B", None)])
    assert infer_game_types(games, {"A": 2, "B": 2}) == ["regular", "cup_final", "regular"]


def test_in_season_tournament_label_2024():
    games = season([("A", "B", "In-Season Tournament"), ("B", "A", None)])
    assert infer_game_types(games, {"A": 1, "B": 1}) == ["cup_final", "regular"]


def test_team_missing_from_standings_never_counts_as_regular():
    games = season([("A", "X", None)])
    assert infer_game_types(games, {"A": 82}) == ["playoff"]
