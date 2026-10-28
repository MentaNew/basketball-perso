"""Career-arc LSTM: predict a player's next-season per-game stats from his previous 4 seasons.

- Split by *target season* (train <= 2016, validation 2017-20, test 2021+), so the model is
  always evaluated on the future, exactly as it is used.
- Inputs include age and season gaps (ml.features.LSTM_INPUT_COLS).
- Every run is scored against three baselines, in real units: Ridge regression on the same
  window, a recency-weighted average, and "same as last season".

Usage: python -m ml.player_lstm train
"""
import argparse
import copy
from functools import cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch import nn

from ml.experiments import log_run
from ml.features import (
    LSTM_INPUT_COLS,
    LSTM_STAT_COLS,
    MAX_GAP,
    SEQ_IN,
    build_career_sequences,
    career_table,
    resolve_player,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
WEIGHTS_PATH = MODELS_DIR / "player_lstm.pt"
META_PATH = MODELS_DIR / "player_lstm_meta.pkl"

HIDDEN_DIM, N_LAYERS, DROPOUT = 64, 1, 0.3   # val sweep: beat 128x2 (0.3493 vs 0.3526 normalised MAE)
LR, WEIGHT_DECAY, EPOCHS, BATCH_SIZE, PATIENCE = 1e-3, 1e-5, 80, 64, 15
TRAIN_MAX_SEASON, VAL_MAX_SEASON = 2016, 2020
STAT_IDX = [LSTM_INPUT_COLS.index(c) for c in LSTM_STAT_COLS]  # where the targets sit in the inputs


def get_device() -> torch.device:
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CareerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                  nn.Linear(hidden_dim // 2, output_dim))

    def forward(self, x):
        out, _ = self.lstm(x)
        # Residual: start from last season's stats and learn only the *change*, which is
        # far easier than predicting the absolute values.
        return x[:, -1, STAT_IDX] + self.head(self.dropout(out[:, -1, :]))


# ---------------------------------------------------------------- baselines (real units)

def baseline_last_season(X: np.ndarray) -> np.ndarray:
    return X[:, -1, STAT_IDX]


def baseline_weighted(X: np.ndarray, weights=(2, 3, 4, 5)) -> np.ndarray:
    """Marcel-style recency-weighted average of the 4 input seasons (oldest -> newest)."""
    w = np.array(weights, dtype=np.float32)[:, None]
    return (X[:, :, STAT_IDX] * w).sum(1) / w.sum()


def mae_table(y: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """MAE per stat in real units, plus `mean_norm_mae` (each stat's MAE / its std, averaged)."""
    table = pd.DataFrame({name: np.abs(p - y).mean(0) for name, p in predictions.items()},
                         index=LSTM_STAT_COLS).T
    table["mean_norm_mae"] = (np.abs(np.stack(list(predictions.values())) - y) / y.std(0)).mean((1, 2))
    return table


# ---------------------------------------------------------------- training

class Normalizer:
    """z-scores inputs and targets with statistics from the *training* windows only."""

    def __init__(self, X_train: np.ndarray):
        flat = X_train.reshape(-1, X_train.shape[2])
        self.x_mean, self.x_std = flat.mean(0), flat.std(0) + 1e-6
        self.y_mean, self.y_std = self.x_mean[STAT_IDX], self.x_std[STAT_IDX]

    def x(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std

    def y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def y_back(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.y_std + self.y_mean


def fit_lstm(X_train, y_train, X_val, y_val, device) -> tuple[CareerLSTM, float]:
    """Train on normalised arrays with early stopping; return the best-on-validation model."""
    model = CareerLSTM(len(LSTM_INPUT_COLS), HIDDEN_DIM, N_LAYERS, len(LSTM_STAT_COLS), DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
    loss_fn = nn.MSELoss()
    X_train, y_train, X_val, y_val = (torch.tensor(a, dtype=torch.float32, device=device)
                                      for a in (X_train, y_train, X_val, y_val))

    best_loss, best_state, epochs_without_gain = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(X_train), device=device)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[batch]), y_train[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        if np.isnan(val_loss):
            raise RuntimeError("validation loss is NaN")
        scheduler.step(val_loss)
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, epochs_without_gain = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_without_gain += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}  val_mse(norm) {val_loss:.4f}  best {best_loss:.4f}")
        if epochs_without_gain >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model.eval(), best_loss


def train() -> pd.DataFrame:
    torch.manual_seed(42)
    np.random.seed(42)
    windows = build_career_sequences(career_table())
    X, y, target_season = windows["X"], windows["y"], windows["meta"]["target_season"].to_numpy()
    train_rows = target_season <= TRAIN_MAX_SEASON
    val_rows = (target_season > TRAIN_MAX_SEASON) & (target_season <= VAL_MAX_SEASON)
    test_rows = target_season > VAL_MAX_SEASON
    print(f"Windows: train {train_rows.sum():,} (≤{TRAIN_MAX_SEASON}), val {val_rows.sum():,}, "
          f"test {test_rows.sum():,} (>{VAL_MAX_SEASON})")

    norm = Normalizer(X[train_rows])
    device = get_device()
    model, best_val_loss = fit_lstm(norm.x(X[train_rows]), norm.y(y[train_rows]),
                                    norm.x(X[val_rows]), norm.y(y[val_rows]), device)

    def lstm_predict(rows):
        with torch.no_grad():
            out = model(torch.tensor(norm.x(X[rows]), dtype=torch.float32, device=device))
        return norm.y_back(out.cpu().numpy())

    flat = lambda rows: norm.x(X[rows]).reshape(rows.sum(), -1)  # noqa: E731  (window -> 1 row)
    ridge = Ridge(alpha=1.0).fit(flat(train_rows), y[train_rows])

    def predictions(rows):
        lstm, linear = lstm_predict(rows), ridge.predict(flat(rows))
        return {"lstm": lstm, "ridge": linear, "blend": (lstm + linear) / 2,
                "weighted_avg": baseline_weighted(X[rows]), "last_season": baseline_last_season(X[rows])}

    # serve whichever model is best on the validation seasons; the test seasons only report
    candidates = ["lstm", "ridge", "blend"]
    served = mae_table(y[val_rows], predictions(val_rows)).loc[candidates, "mean_norm_mae"].idxmin()
    report = mae_table(y[test_rows], predictions(test_rows))
    print(f"\nTest MAE in real units (target seasons > {VAL_MAX_SEASON}). Served model: {served}")
    print(report.round(4).to_string())

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.cpu().state_dict(), WEIGHTS_PATH)
    joblib.dump({
        "input_cols": LSTM_INPUT_COLS, "stat_cols": LSTM_STAT_COLS,
        "x_mean": norm.x_mean, "x_std": norm.x_std, "y_mean": norm.y_mean, "y_std": norm.y_std,
        "best_val_loss": best_val_loss, "test_report": report, "served": served, "ridge": ridge,
        "hparams": {"hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "dropout": DROPOUT},
    }, META_PATH)
    log_run("player_lstm", metrics={"best_val_loss": best_val_loss,
                                    **report["mean_norm_mae"].add_prefix("test_norm_mae_").to_dict()},
            params={"hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "dropout": DROPOUT,
                    "inputs": LSTM_INPUT_COLS, "served": served})
    print(f"\nSaved {WEIGHTS_PATH} and {META_PATH}")
    return report


# ---------------------------------------------------------------- inference

@cache
def load_bundle() -> dict:
    # xgboost and torch ship separate OpenMP runtimes on macOS; once xgboost has run in the
    # same process (the Streamlit app), multi-threaded torch ops can deadlock.
    torch.set_num_threads(1)
    meta = joblib.load(META_PATH)
    hp = meta["hparams"]
    model = CareerLSTM(len(meta["input_cols"]), hp["hidden_dim"], hp["n_layers"],
                       len(meta["stat_cols"]), hp["dropout"])
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    return {"model": model.eval(), "meta": meta, "careers": career_table()}


def predict_next_season(player: str) -> dict | None:
    """Projection for the season after the player's latest one. None if he has fewer than 4
    qualifying (20+ games) seasons, or the last 4 contain a gap of more than one season."""
    bundle = load_bundle()
    model, meta, careers = bundle["model"], bundle["meta"], bundle["careers"]
    bref_id = resolve_player(careers, player)
    if bref_id is None:
        return None
    history = careers[careers.bref_id == bref_id].tail(SEQ_IN)
    if len(history) < SEQ_IN or (history["gap"].iloc[1:] > MAX_GAP).any():
        return None

    window = (history[meta["input_cols"]].to_numpy(np.float32)[None] - meta["x_mean"]) / meta["x_std"]
    with torch.no_grad():
        lstm = model(torch.tensor(window, dtype=torch.float32)).numpy()[0] * meta["y_std"] + meta["y_mean"]
    linear = meta["ridge"].predict(window.reshape(1, -1))[0]
    prediction = {"lstm": lstm, "ridge": linear, "blend": (lstm + linear) / 2}[meta.get("served", "lstm")]
    return {
        "player": history["name"].iat[0],
        "based_on_seasons": history["season"].tolist(),
        "target_season": int(history["season"].iat[-1]) + 1,
        "age_next": int(history["age"].iat[-1]) + 1,
        "model": meta.get("served", "lstm"),
        "predicted_next_season": dict(zip(meta["stat_cols"], map(float, prediction), strict=True)),
        "last_season": dict(zip(meta["stat_cols"], map(float, history[meta["stat_cols"]].iloc[-1]),
                                strict=True)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the career-arc LSTM.")
    parser.add_argument("command", choices=["train"])
    parser.parse_args()
    train()
