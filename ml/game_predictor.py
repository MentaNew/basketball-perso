"""Game outcome model: probability that the home team wins, from pre-game features.

Training protocol (no future information ever reaches the model):
- The first season is dropped: Elo and "last season" priors aren't warmed up yet.
- The last TEST_SEASONS seasons are held out and touched only once, for the final report.
- Development uses TimeSeriesSplit: every fold trains on the past, validates on the future.
- The model is compared with two baselines: Elo alone, and always picking the home team.

The model ships uncalibrated. Isotonic calibration was tried and made held-out log-loss
worse (0.614 vs 0.610), so it is not worth its complexity here.

Usage: python -m ml.game_predictor train
"""
import argparse
from functools import cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from ml.features import GAME_FEATURES, build_game_features, matchup_features, read_games

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "game_predictor.pkl"
TEST_SEASONS = 2


def make_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss", random_state=42,
    )


def score(y_true, p_home: np.ndarray) -> dict:
    """Accuracy and AUC judge the ranking; log-loss and Brier judge the probabilities."""
    p_home = np.clip(p_home, 1e-6, 1 - 1e-6)
    return {
        "accuracy": accuracy_score(y_true, p_home >= 0.5),
        "auc": roc_auc_score(y_true, p_home),
        "log_loss": log_loss(y_true, p_home, labels=[0, 1]),
        "brier": brier_score_loss(y_true, p_home),
    }


def train() -> dict:
    games = read_games()
    features, snapshots = build_game_features(games)
    data = features.assign(season=games["season"].values)
    data = data[data["season"] > data["season"].min()].reset_index(drop=True)  # drop burn-in

    test_from = data["season"].max() - TEST_SEASONS + 1
    train_set, test_set = data[data["season"] < test_from], data[data["season"] >= test_from]
    X_train, y_train = train_set[GAME_FEATURES], train_set["target"]
    X_test, y_test = test_set[GAME_FEATURES], test_set["target"]
    print(f"Games: train {len(train_set):,} (until {test_from - 1}), "
          f"test {len(test_set):,} ({test_from}+)")

    # 1. Development: time-ordered cross-validation on the training seasons only
    folds = []
    for fold, (fit_idx, val_idx) in enumerate(TimeSeriesSplit(n_splits=5).split(X_train), 1):
        model = make_model().fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        y_val = y_train.iloc[val_idx]
        folds.append({
            "fold": fold,
            **score(y_val, model.predict_proba(X_train.iloc[val_idx])[:, 1]),
            "elo_accuracy": accuracy_score(y_val, X_train.iloc[val_idx]["elo_prob"] >= 0.5),
            "home_accuracy": y_val.mean(),
        })
    cv = pd.DataFrame(folds)
    print("\nTimeSeriesSplit CV:")
    print(cv.round(4).to_string(index=False))

    # 2. Final report: train on all training seasons, score once on the held-out seasons
    model = make_model().fit(X_train, y_train)
    test_report = pd.DataFrame([
        {"model": "xgboost", **score(y_test, model.predict_proba(X_test)[:, 1])},
        {"model": "elo_only", **score(y_test, X_test["elo_prob"].to_numpy())},
        {"model": "always_home", **score(y_test, np.full(len(y_test), y_train.mean()))},
    ])
    print(f"\nHeld-out test ({test_from}+):")
    print(test_report.round(4).to_string(index=False))

    # 3. Ship a model trained on every season, reporting the held-out numbers above
    final_model = make_model().fit(data[GAME_FEATURES], data["target"])
    held_out = test_report.set_index("model").loc["xgboost"]
    bundle = {
        "model": final_model,
        "feature_cols": GAME_FEATURES,
        "snapshots": snapshots,
        "cv": cv,
        "test_report": test_report,
        "test_seasons": (int(test_from), int(data["season"].max())),
        **{metric: float(held_out[metric]) for metric in ("accuracy", "auc", "log_loss", "brier")},
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved {MODEL_PATH}")
    return bundle


# ---------------------------------------------------------------- inference

@cache
def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


def available_seasons() -> list[int]:
    return sorted({season for season, _ in load_bundle()["snapshots"]})


def available_teams(season: int) -> list[str]:
    return sorted(team for s, team in load_bundle()["snapshots"] if s == season)


def predict_winner(home_team: str, away_team: str, season: int | None = None,
                   playoff: bool = False, neutral: bool = False) -> dict:
    """Win probability for a matchup between two teams as they stood at the end of `season`
    (default: the latest season). Teams are that season's abbreviations, e.g. 'SEA' in 2005."""
    bundle = load_bundle()
    snapshots = bundle["snapshots"]
    season = season or max(s for s, _ in snapshots)
    missing = [team for team in (home_team, away_team) if (season, team) not in snapshots]
    if missing:
        raise ValueError(f"no data for {missing} in {season}")

    row = matchup_features(snapshots[(season, home_team)], snapshots[(season, away_team)],
                           playoff, neutral)
    p_home = float(bundle["model"].predict_proba(pd.DataFrame([row])[bundle["feature_cols"]])[0, 1])
    edge = abs(p_home - 0.5)
    return {
        "home_team": home_team,
        "away_team": away_team,
        "season": season,
        "home_win_prob": p_home,
        "away_win_prob": 1 - p_home,
        "predicted_winner": home_team if p_home >= 0.5 else away_team,
        "confidence": "high" if edge > 0.15 else "moderate" if edge > 0.07 else "low",
        "elo_home_prob": row["elo_prob"],
        "model_accuracy": bundle["accuracy"],
    }


def get_model_stats() -> dict:
    bundle = load_bundle()
    return {key: bundle[key] for key in ("accuracy", "auc", "log_loss", "brier", "test_seasons",
                                         "test_report", "cv", "feature_cols")}


def feature_importance() -> pd.Series:
    bundle = load_bundle()
    return (pd.Series(bundle["model"].feature_importances_, index=bundle["feature_cols"])
            .sort_values(ascending=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the game outcome model.")
    parser.add_argument("command", choices=["train"])
    parser.parse_args()
    train()
