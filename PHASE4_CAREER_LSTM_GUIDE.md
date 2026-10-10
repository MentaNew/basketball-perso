# Phase 4 — Career-Arc LSTM

Predict a player's **next-season** per-game stats (`pts, reb, ast, stl, blk, fg_pct`) from
their previous 4 seasons, and measure it honestly against simple baselines.

Files: `ml/features.py` (career-sequence section), `ml/player_lstm.py`, `tests/test_features.py`.

---

## Step 0 — What was wrong in v1 (§7.7)

| v1 problem | Consequence | Fix in this phase |
|---|---|---|
| Random 85/15 split over *windows* | Windows from the same player landed in both train and val, and the model memorised careers | **Temporal split** by target season |
| Consecutive rows assumed to be consecutive seasons | A player returning after 2 missed years looked like a normal progression | `gap` feature + gap rules |
| No age input | Age is the strongest aging-curve signal | `age` feature |
| No baseline | `val_loss = 0.256` meant nothing | 3 baselines, MAE in real units |
| Players keyed by name | Namesakes merged into one career | Fixed in Phase 1 (`bref_id`) |

---

## Step 1 — Career table (`career_table`)

```python
df = read_player_seasons(min_games=20)               # primary rows (TOT for traded players)
df = df.dropna(subset=LSTM_STAT_COLS + ["mp", "age"])
df["gap"] = seasons since the player's previous qualifying season (1 = consecutive, capped at 3)
```

Input features per season (`LSTM_INPUT_COLS`, 10):

| Feature | Why |
|---|---|
| `pts, reb, ast, stl, blk, fg_pct` | What we predict; history is the best predictor |
| `mp` | Minutes drive counting stats. A role change shows up here first |
| `g` | Games played: availability and injury signal |
| `age` | Young players improve and veterans decline, the classic aging curve |
| `gap` | 1 = consecutive. 2 = a missed season (injury, overseas, < 20 games) |

---

## Step 2 — Windows (`build_career_sequences`)

```
seasons:  2016  2017  2018  2019 │ 2020
          └────── X (4×10) ─────┘ └ y (6) ┘
```

Rules:
- `SEQ_IN = 4` seasons in, the **next** season out.
- The target must directly follow the window (`gap == 1`). We predict "next season", not
  "whenever he returns".
- Inside the window, gaps up to `MAX_GAP = 2` (one missed season) are allowed, and the `gap`
  feature tells the model.
- Data is returned **unnormalised**. Normalisation happens in training, using training-set
  statistics only (otherwise the val/test distribution leaks into training).

The result is 7,709 windows. `test_sequences_windows_and_gap_rules` pins all three rules.

---

## Step 3 — Split

```
target season:   ≤ 2016   │  2017–2020  │  2021–2026
                 train     │  validation │  test
                 5,702     │  786        │  1,221
```

Splitting on the **target season** means the model is always tested on the future, which is
exactly how it is used. One player's early windows can sit in train and his late windows in
test. That is legitimate, because it mirrors real use: you know a player's past and predict
his future.

Normalisation: `x_mean` and `x_std` are computed on training windows. Target stats use the same
mean/std as their input columns (`y_mean = x_mean[STAT_IDX]`).

---

## Step 4 — Model (`CareerLSTM`)

```python
class CareerLSTM(nn.Module):
    lstm = nn.LSTM(10, 128, num_layers=2, batch_first=True, dropout=0.3)
    head = Linear(128, 64) → ReLU → Linear(64, 6)

    def forward(self, x):
        out, _ = self.lstm(x)
        return x[:, -1, STAT_IDX] + self.head(self.dropout(out[:, -1, :]))
        #      └─ last season ─┘   └────── learned change ──────┘
```

**The residual connection is the key design choice.** The network predicts the *change* from
last season rather than the absolute value. "Same as last year" is already a strong
predictor, so starting there means the network only has to learn the aging and role-change
adjustments.

Training (same hyperparameters as v1 where they still apply):

| Setting | Value |
|---|---|
| Optimiser | Adam, `lr=1e-3`, `weight_decay=1e-5` |
| Loss | MSE on normalised targets |
| Batch / epochs | 64 / up to 80 |
| LR schedule | `ReduceLROnPlateau(patience=8, factor=0.5)` |
| Early stopping | 15 epochs without val improvement; the best state dict is restored |
| Gradient clipping | 1.0 |
| Device | mps → cuda → cpu |

---

## Step 5 — Baselines and evaluation

Every training run prints MAE **in real units** on the test windows for four predictors:

| Predictor | How |
|---|---|
| `lstm` | The model |
| `ridge` | `Ridge(alpha=1)` on the same 4×10 window, flattened to 40 numbers |
| `weighted_avg` | Marcel-style recency weights 2/3/4/5 (oldest → newest) |
| `last_season` | Copy last season |

`mean_norm_mae` divides each stat's MAE by its standard deviation and averages, so all six
stats count equally.

### Results (data through 2025-26)

| model | pts | reb | ast | stl | blk | fg% | mean_norm_mae |
|---|---|---|---|---|---|---|---|
| LSTM | 2.23 | 0.78 | 0.64 | 0.181 | 0.152 | 0.031 | 0.364 |
| **Ridge** | **2.19** | **0.76** | **0.62** | **0.173** | **0.140** | **0.030** | **0.350** |
| weighted avg | 2.75 | 0.91 | 0.71 | 0.189 | 0.150 | 0.031 | 0.391 |
| last season | 2.36 | 0.83 | 0.67 | 0.195 | 0.157 | 0.034 | 0.386 |

**The honest read:** the LSTM beats the naive baselines but **loses to a linear model**. With
~6k training windows and smooth year-over-year dynamics, the sequence model has no
non-linear structure to exploit that ridge can't already capture. That isn't a failure of
the phase, because finding it is the reason the baselines exist. See `NEXT_STEPS.md` for ideas
to make the deep model earn its place.

---

## Step 6 — Artifacts

| File | Contents |
|---|---|
| `models/player_lstm.pt` | `state_dict` |
| `models/player_lstm_meta.pkl` | `input_cols, stat_cols, x_mean, x_std, y_mean, y_std, best_val_loss, test_report, hparams` |

The meta file contains everything needed to rebuild the network and denormalise its output,
so inference never re-derives statistics from the database.

---

## Step 7 — Inference API

```python
from ml.player_lstm import predict_next_season
predict_next_season("LeBron James")
# {'player': 'LeBron James', 'based_on_seasons': [2023, 2024, 2025, 2026],
#  'target_season': 2027, 'age_next': 42,
#  'predicted_next_season': {'pts': 19.0, 'reb': 5.8, 'ast': 6.6, ...},
#  'last_season':           {'pts': 20.9, 'reb': 6.1, 'ast': 7.2, ...}}
```

It returns `None` when the player has fewer than 4 qualifying seasons (g ≥ 20) or a gap of
more than one missed season in the window. Example: Wembanyama has only 2 seasons.

`load_bundle()` sets `torch.set_num_threads(1)` because of the macOS OpenMP clash with xgboost in
the same process (see Phase 2).

---

## Step 8 — Run & verify

```bash
export PYTHONPATH=.
python -m ml.player_lstm train     # ~1 min (early-stops around epoch 45)
```

Checks:
- [ ] The LSTM beats `last_season` on every stat. If not, training broke.
- [ ] Compare the LSTM against `ridge` and note honestly which one wins.
- [ ] Projections for old players (age 35+) trend down and for young players (≤ 23) trend up.

---

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| `RuntimeError: validation loss is NaN` | Bad input row or exploding gradients | Check `career_table()` for NaN or inf; lower the LR |
| Process hangs at the first prediction inside the app | OpenMP clash | Keep `set_num_threads(1)` in `load_bundle()` |
| Val loss differs between runs on MPS | Non-deterministic GPU kernels | Train with `device=cpu` for exact reproducibility |
| Projection is `None` for an active player | < 4 seasons with 20+ games | Expected; the UI explains it |

---

## What Phase 5 Expects From This

`models/player_lstm.pt` and `models/player_lstm_meta.pkl`, plus `predict_next_season(player)`.
The Player Profile page's projection tab is the only consumer.
