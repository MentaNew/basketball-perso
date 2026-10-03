"""Game outcome model: XGBoost + isotonic calibration on leak-free pre-game features.

Usage: python -m ml.game_predictor train
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from ml.features import GAME_FEATURES, build_game_features, load_games, matchup_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "game_predictor.pkl"
BURN_IN_SEASONS = 1      # first season: Elo and "previous season" priors aren't warmed up yet
TEST_SEASONS = 2         # most recent seasons held out for the final report


def make_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss", random_state=42,
    )


def metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "accuracy": accuracy_score(y, p >= 0.5),
        "auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
    }


def evaluate(name: str, y, p, table: list):
    m = metrics(y, p)
    table.append({"model": name, **m})


def train():
    games = load_games()
    feats, snapshots = build_game_features(games)
    data = feats.assign(season=games["season"].values)
    first = data["season"].min() + BURN_IN_SEASONS
    data = data[data["season"] >= first].reset_index(drop=True)
    test_from = data["season"].max() - TEST_SEASONS + 1
    train_df, test_df = data[data.season < test_from], data[data.season >= test_from]
    X_tr, y_tr = train_df[GAME_FEATURES], train_df["target"]
    X_te, y_te = test_df[GAME_FEATURES], test_df["target"]
    print(f"Games: train {len(train_df):,} ({first}–{test_from - 1}), "
          f"test {len(test_df):,} ({test_from}–{data.season.max()})")

    # --- time-aware CV on the training seasons
    cv_rows = []
    for fold, (tr, va) in enumerate(TimeSeriesSplit(n_splits=5).split(X_tr), 1):
        model = make_xgb().fit(X_tr.iloc[tr], y_tr.iloc[tr])
        p = model.predict_proba(X_tr.iloc[va])[:, 1]
        y = y_tr.iloc[va]
        cv_rows.append({"fold": fold, **metrics(y, p),
                        "elo_acc": accuracy_score(y, X_tr.iloc[va]["elo_prob"] >= 0.5),
                        "home_acc": y.mean()})
    cv = pd.DataFrame(cv_rows)
    print("\nTimeSeriesSplit CV (XGBoost vs. Elo-only vs. always-home accuracy):")
    print(cv.round(4).to_string(index=False))

    # --- calibrated model, scored on held-out recent seasons against baselines
    calibrated = CalibratedClassifierCV(make_xgb(), cv=TimeSeriesSplit(n_splits=5),
                                        method="isotonic").fit(X_tr, y_tr)
    home_rate = y_tr.mean()
    report = []
    evaluate("xgb_calibrated", y_te, calibrated.predict_proba(X_te)[:, 1], report)
    evaluate("xgb_raw", y_te, make_xgb().fit(X_tr, y_tr).predict_proba(X_te)[:, 1], report)
    evaluate("elo_only", y_te, X_te["elo_prob"].values, report)
    evaluate("always_home", y_te, np.full(len(y_te), home_rate), report)
    test_report = pd.DataFrame(report)
    print(f"\nHeld-out test ({test_from}+):")
    print(test_report.round(4).to_string(index=False))

    # --- final model on everything
    final = CalibratedClassifierCV(make_xgb(), cv=TimeSeriesSplit(n_splits=5),
                                   method="isotonic").fit(data[GAME_FEATURES], data["target"])
    xgb_test = test_report.set_index("model").loc["xgb_calibrated"]
    bundle = {
        "model": final,
        "feature_cols": GAME_FEATURES,
        "snapshots": snapshots,
        "cv": cv,
        "test_report": test_report,
        "test_seasons": (int(test_from), int(data.season.max())),
        "accuracy": float(xgb_test["accuracy"]),
        "auc": float(xgb_test["auc"]),
        "log_loss": float(xgb_test["log_loss"]),
        "brier": float(xgb_test["brier"]),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved {MODEL_PATH}")
    return bundle


# ---------------------------------------------------------------- inference

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def available_seasons() -> list[int]:
    return sorted({s for s, _ in _load()["snapshots"]})


def available_teams(season: int) -> list[str]:
    return sorted(t for s, t in _load()["snapshots"] if s == season)


def predict_winner(home_team: str, away_team: str, season: int | None = None,
                   playoff: bool = False) -> dict:
    """Win probability for a hypothetical matchup, using each team's end-of-season state
    (Elo, form, net rating) for `season` (default: latest)."""
    b = _load()
    snaps = b["snapshots"]
    if season is None:
        season = max(s for s, _ in snaps)
    missing = [t for t in (home_team, away_team) if (season, t) not in snaps]
    if missing:
        raise ValueError(f"no data for {missing} in {season}")
    row = matchup_features(snaps[(season, home_team)], snaps[(season, away_team)], playoff)
    X = pd.DataFrame([row])[b["feature_cols"]]
    p = float(b["model"].predict_proba(X)[0, 1])
    margin = abs(p - 0.5)
    return {
        "home_team": home_team,
        "away_team": away_team,
        "season": season,
        "home_win_prob": p,
        "away_win_prob": 1 - p,
        "predicted_winner": home_team if p >= 0.5 else away_team,
        "confidence": "high" if margin > 0.15 else "moderate" if margin > 0.07 else "low",
        "elo_home_prob": row["elo_prob"],
        "model_accuracy": b["accuracy"],
    }


def get_model_stats() -> dict:
    b = _load()
    return {k: b[k] for k in ("accuracy", "auc", "log_loss", "brier", "test_seasons",
                              "test_report", "cv", "feature_cols")}


def feature_importance() -> pd.Series:
    b = _load()
    imps = [c.estimator.feature_importances_ for c in b["model"].calibrated_classifiers_]
    return pd.Series(np.mean(imps, axis=0), index=b["feature_cols"]).sort_values(ascending=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.parse_args()
    train()
