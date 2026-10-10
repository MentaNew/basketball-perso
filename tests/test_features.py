"""Feature tests that need no database: leakage, Elo, sequences, archetype naming."""
import numpy as np
import pandas as pd

from etl.teams import franchise_of
from ml import features as F


def toy_games() -> pd.DataFrame:
    rows = [
        # season, date, away, home, away_score, home_score
        (2001, "2000-11-01", "BOS", "SEA", 90, 100),
        (2001, "2000-11-02", "SEA", "BOS", 95, 99),
        (2001, "2000-11-03", "BOS", "SEA", 80, 120),
        (2002, "2001-11-01", "BOS", "SEA", 101, 100),
        (2002, "2001-11-02", "OKC", "BOS", 70, 110),   # relocated franchise, same state
    ]
    df = pd.DataFrame(rows, columns=["season", "game_date", "away_team", "home_team",
                                     "away_score", "home_score"])
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["home_win"] = df.home_score > df.away_score
    df["game_type"] = "regular"
    df["neutral"] = False
    return df


def test_features_use_only_past_games():
    games = toy_games()
    feats, _ = F.build_game_features(games)
    # first game: nobody has played; everything is the prior
    first = feats.iloc[0]
    assert first.home_gp == 0 and first.home_elo == F.ELO_BASE and first.elo_diff == 0
    # changing a game's result must not change that game's own features (no leakage)
    flipped = games.copy()
    flipped.loc[2, ["away_score", "home_score", "home_win"]] = [130, 80, False]
    feats2, _ = F.build_game_features(flipped)
    pd.testing.assert_series_equal(feats.iloc[2].drop("target"), feats2.iloc[2].drop("target"))
    assert feats.iloc[2].target != feats2.iloc[2].target
    # ...but it does change later games
    assert feats.iloc[3].home_elo != feats2.iloc[3].home_elo


def test_season_rollover_and_relocation():
    feats, snaps = F.build_game_features(toy_games())
    g = feats.iloc[3]                                 # 2002 opener, SEA at home
    assert g.home_gp == 0 and g.home_prev_w_pct == 2 / 3
    end_2001 = snaps[(2001, "SEA")]["elo"]
    assert np.isclose(g.home_elo, F.ELO_CARRY * end_2001 + (1 - F.ELO_CARRY) * F.ELO_BASE)
    assert feats.iloc[4].away_gp == 1                 # OKC continues SEA's 2002 season
    assert franchise_of("SEA") == "OKC"
    assert (2001, "SEA") in snaps and (2002, "OKC") in snaps


def test_rest_days_and_back_to_back():
    feats, _ = F.build_game_features(toy_games())
    assert feats.iloc[1].home_rest == 1 and feats.iloc[1].home_b2b == 1
    assert feats.iloc[0].home_rest == F.REST_CAP


def test_elo_update_is_zero_sum_and_directional():
    h, a = F.elo_update(1500, 1500, home_margin=10)
    assert h > 1500 > a and np.isclose(h + a, 3000)
    assert F.elo_win_prob(1500, 1500) > 0.5          # home-court advantage


def career_df(seasons, gaps=None):
    n = len(seasons)
    df = pd.DataFrame({
        "bref_id": "x01", "name": "X", "season": seasons,
        **{c: np.arange(n, dtype=float) + i for i, c in enumerate(F.LSTM_STAT_COLS)},
        "mp": 30.0, "g": 70.0, "age": np.arange(22, 22 + n, dtype=float),
    })
    df["gap"] = df["season"].diff().fillna(1)
    return df


def test_sequences_windows_and_gap_rules():
    seq = F.build_career_sequences(career_df([2001, 2002, 2003, 2004, 2005, 2006]))
    assert seq["X"].shape == (2, F.SEQ_IN, len(F.LSTM_INPUT_COLS))
    assert list(seq["meta"].target_season) == [2005, 2006]
    # target must directly follow the window
    seq = F.build_career_sequences(career_df([2001, 2002, 2003, 2004, 2006]))
    assert len(seq["X"]) == 0
    # a 2-year jump inside the window is allowed (gap feature = 2)
    seq = F.build_career_sequences(career_df([2001, 2003, 2004, 2005, 2006]))
    assert len(seq["X"]) == 1 and seq["X"][0, 1, F.LSTM_INPUT_COLS.index("gap")] == 2


def test_archetype_names_follow_centroids():
    from ml.similarity import ARCHETYPE_TEMPLATES, name_clusters
    names = ["Traditional Big", "Floor General", "3-and-D Wing"]
    cents = pd.DataFrame(
        [[ARCHETYPE_TEMPLATES[n].get(c, 0.0) for c in F.SIMILARITY_COLS] for n in names],
        columns=F.SIMILARITY_COLS, index=[7, 3, 0])   # cluster ids deliberately scrambled
    assert name_clusters(cents) == {7: "Traditional Big", 3: "Floor General", 0: "3-and-D Wing"}


def test_neutral_games_have_no_home_court():
    games = toy_games()
    games.loc[0, "neutral"] = True
    feats, _ = F.build_game_features(games)
    assert feats.iloc[0].is_neutral == 1 and feats.iloc[1].is_neutral == 0
    assert feats.iloc[0].elo_prob == 0.5               # equal ratings, no home edge
    # the update also drops home court: the home winner gains more than with HCA
    with_hca, _ = F.build_game_features(games.assign(neutral=False))
    assert feats.iloc[1].away_elo > with_hca.iloc[1].away_elo   # SEA won game 0 at "home"
