"""Symmetry of the game model: no database needed."""
import numpy as np
import pandas as pd

from ml import game_predictor as GP
from ml.features import GAME_FEATURES


def toy_games(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(GAME_FEATURES))), columns=GAME_FEATURES)
    X["elo_prob"] = rng.uniform(0.2, 0.8, n)
    X["home_court"] = 1
    y = pd.Series((rng.uniform(size=n) < X["elo_prob"]).astype(int))
    return X, y


def test_mirroring_twice_gives_the_original():
    X, _ = toy_games(10)
    pd.testing.assert_frame_equal(GP.mirror(GP.mirror(X)), X)


def test_predictions_are_symmetric_and_neutral_equal_teams_are_a_coin_flip():
    X, y = toy_games()
    model = GP.fit(X, y)
    p, p_mirrored = GP.predict_home(model, X), GP.predict_home(model, GP.mirror(X))
    np.testing.assert_allclose(p + p_mirrored, 1.0)                     # both sides add up to 1
    equal_neutral = GP.mirror(X.iloc[:1]).iloc[:1] * 0                  # identical teams...
    equal_neutral = equal_neutral.assign(elo_prob=0.5, home_court=0)     # ...at a neutral site
    assert GP.predict_home(model, equal_neutral)[0] == 0.5
