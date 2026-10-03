"""Career-arc LSTM: predict next-season per-game stats from the previous 4 seasons.

Split is temporal (by target season), so no player-window leaks across train/val/test
(§7.7).  Inputs include age and season gaps.  Every run is compared to baselines.

Usage: python -m ml.player_lstm train
"""
import argparse
import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch import nn

from ml.features import (
    LSTM_INPUT_COLS,
    LSTM_STAT_COLS,
    SEQ_IN,
    build_career_sequences,
    career_table,
    resolve_player,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
WEIGHTS_PATH = MODELS_DIR / "player_lstm.pt"
META_PATH = MODELS_DIR / "player_lstm_meta.pkl"

HIDDEN_DIM, N_LAYERS, DROPOUT = 128, 2, 0.3
LR, WEIGHT_DECAY, EPOCHS, BATCH_SIZE, PATIENCE = 1e-3, 1e-5, 80, 64, 15
TRAIN_MAX_SEASON, VAL_MAX_SEASON = 2016, 2020   # test = target seasons after 2020
STAT_IDX = [LSTM_INPUT_COLS.index(c) for c in LSTM_STAT_COLS]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
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
        # residual on the last season: the net learns the *change*, which is easier
        return x[:, -1, STAT_IDX] + self.head(self.dropout(out[:, -1, :]))


# ---------------------------------------------------------------- baselines (real units)

def baseline_last_season(X: np.ndarray) -> np.ndarray:
    return X[:, -1, STAT_IDX]


def baseline_weighted(X: np.ndarray, weights=(2, 3, 4, 5)) -> np.ndarray:
    """Marcel-style recency-weighted average of the input seasons (oldest -> newest)."""
    w = np.array(weights, dtype=np.float32)[:, None]
    return (X[:, :, STAT_IDX] * w).sum(1) / w.sum()


def mae_table(y: np.ndarray, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = {name: np.abs(p - y).mean(0) for name, p in preds.items()}
    t = pd.DataFrame(rows, index=LSTM_STAT_COLS).T
    t["mean_norm_mae"] = (np.abs(np.stack(list(preds.values())) - y) / y.std(0)).mean((1, 2))
    return t


# ---------------------------------------------------------------- training

def train():
    torch.manual_seed(42)
    np.random.seed(42)
    seq = build_career_sequences(career_table())
    X, y, meta = seq["X"], seq["y"], seq["meta"]
    tr = (meta.target_season <= TRAIN_MAX_SEASON).values
    va = ((meta.target_season > TRAIN_MAX_SEASON) & (meta.target_season <= VAL_MAX_SEASON)).values
    te = (meta.target_season > VAL_MAX_SEASON).values
    print(f"Windows: train {tr.sum():,} (≤{TRAIN_MAX_SEASON}), val {va.sum():,}, "
          f"test {te.sum():,} (>{VAL_MAX_SEASON})")

    x_mean, x_std = X[tr].reshape(-1, X.shape[2]).mean(0), X[tr].reshape(-1, X.shape[2]).std(0) + 1e-6
    y_mean, y_std = x_mean[STAT_IDX], x_std[STAT_IDX]
    norm_x = lambda a: (a - x_mean) / x_std  # noqa: E731
    norm_y = lambda a: (a - y_mean) / y_std  # noqa: E731

    device = get_device()
    model = CareerLSTM(len(LSTM_INPUT_COLS), HIDDEN_DIM, N_LAYERS, len(LSTM_STAT_COLS),
                       DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    loss_fn = nn.MSELoss()
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    Xtr, ytr, Xva, yva = t(norm_x(X[tr])), t(norm_y(y[tr])), t(norm_x(X[va])), t(norm_y(y[va]))

    best, best_state, stale = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(perm), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva), yva).item()
        sched.step(val)
        if np.isnan(val):
            raise RuntimeError("validation loss is NaN")
        if val < best - 1e-5:
            best, best_state, stale = val, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}  val_mse(norm) {val:.4f}  best {best:.4f}")
        if stale >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break
    model.load_state_dict(best_state)
    model.eval()

    def predict(a):
        with torch.no_grad():
            return model(t(norm_x(a))).cpu().numpy() * y_std + y_mean

    ridge = Ridge(alpha=1.0).fit(norm_x(X[tr]).reshape(tr.sum(), -1), y[tr])
    preds = {
        "lstm": predict(X[te]),
        "ridge": ridge.predict(norm_x(X[te]).reshape(te.sum(), -1)),
        "weighted_avg": baseline_weighted(X[te]),
        "last_season": baseline_last_season(X[te]),
    }
    report = mae_table(y[te], preds)
    print(f"\nTest MAE in real units (target seasons > {VAL_MAX_SEASON}):")
    print(report.round(4).to_string())

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(model.cpu().state_dict(), WEIGHTS_PATH)
    joblib.dump({
        "input_cols": LSTM_INPUT_COLS, "stat_cols": LSTM_STAT_COLS,
        "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
        "best_val_loss": best, "test_report": report,
        "hparams": {"hidden_dim": HIDDEN_DIM, "n_layers": N_LAYERS, "dropout": DROPOUT},
    }, META_PATH)
    print(f"\nSaved {WEIGHTS_PATH} and {META_PATH}")
    return report


# ---------------------------------------------------------------- inference

_cache: dict = {}


def _load():
    if not _cache:
        # xgboost and torch ship separate OpenMP runtimes on macOS; once xgboost has run in
        # the same process (the Streamlit app), multi-threaded torch ops can deadlock.
        torch.set_num_threads(1)
        meta = joblib.load(META_PATH)
        hp = meta["hparams"]
        model = CareerLSTM(len(meta["input_cols"]), hp["hidden_dim"], hp["n_layers"],
                           len(meta["stat_cols"]), hp["dropout"])
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        model.eval()
        _cache.update(model=model, meta=meta, careers=career_table())
    return _cache


def predict_next_season(player: str) -> dict | None:
    """Projection for the season after the player's latest one.  None if fewer than 4
    qualifying (g ≥ 20) seasons, or a gap > 1 missed season inside the window."""
    c = _load()
    careers, meta, model = c["careers"], c["meta"], c["model"]
    bref_id = resolve_player(careers, player)
    if bref_id is None:
        return None
    hist = careers[careers.bref_id == bref_id].tail(SEQ_IN)
    if len(hist) < SEQ_IN or (hist["gap"].iloc[1:] > 2).any():
        return None
    x = hist[meta["input_cols"]].to_numpy(np.float32)[None]
    with torch.no_grad():
        p = model(torch.tensor((x - meta["x_mean"]) / meta["x_std"])).numpy()[0]
    pred = p * meta["y_std"] + meta["y_mean"]
    return {
        "player": hist["name"].iat[0],
        "based_on_seasons": hist["season"].tolist(),
        "target_season": int(hist["season"].iat[-1]) + 1,
        "age_next": int(hist["age"].iat[-1]) + 1,
        "predicted_next_season": dict(zip(meta["stat_cols"], map(float, pred), strict=True)),
        "last_season": dict(zip(meta["stat_cols"],
                                map(float, hist[meta["stat_cols"]].iloc[-1]), strict=True)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.parse_args()
    train()
