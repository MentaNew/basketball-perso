# Phase 2 — Game Predictor

Predict the probability that the home team wins an NBA game, using only information
available **before tip-off**, and prove the model beats simple baselines on seasons it never saw.

Files: `ml/features.py` (game-feature section), `ml/game_predictor.py`, `tests/test_features.py`.

---

## Step 0 — Why this phase was rebuilt

The v1 model reported good accuracy, but it was invalid for two reasons (blueprint §7.1 and §7.2):

1. **Inverted labels.** Home and away scores were swapped in the database, so the model
   learned "home team loses".
2. **Target leakage.** Its features were each team's *final* standings for the same season
   (`w_pct`, `ps_g`, `pa_g`). Those numbers already include the game being predicted and every
   later game. It is like predicting a game from the season's final table.

Phase 1 fixed the labels. This phase fixes the leakage. The rule to keep in mind:

> A feature for game *g* may only use games that finished **before** *g* started.

---

## Step 1 — Inputs

```python
from ml.features import load_games
games = load_games()   # regular + play_in + playoff, ordered by game_date, game_id
```

`load_games()` returns rows from the `games` table. The Phase 1 loader sets each game's
`game_type`: a team's first W+L games are its regular season, BRef marks play-in games, and
the season's last "NBA Cup" or "In-Season Tournament" game is the non-counting `cup_final`.
Cup finals are excluded by default.

Only three columns of a game matter to the model:
- **Label:** `home_win`.
- **State update:** `home_score - away_score` (the margin).
- **Context:** `neutral`, which is true for games abroad and the 2020 bubble (set by the
  Phase 1 parser).

Everything else is derived from earlier games.

---

## Step 2 — Feature engineering (`build_game_features`)

### 2.1 The walk-forward loop

The whole design is one chronological pass over ~55k games:

```
for each game g (oldest → newest):
    1. read the current state of both teams → features for g      (pre-game)
    2. append features
    3. update both teams' state with g's result                     (post-game)
```

Step 1 always happens before step 3, so leakage is impossible by construction. That is
easier to trust than a pile of `groupby().shift(1)` calls.

### 2.2 Team state (`TeamState`)

State is keyed by **franchise** (`etl/teams.py`: `SEA→OKC`, `NJN→BRK`, `VAN→MEM`, …), so a
relocated team keeps its rating.

| Field | Meaning |
|---|---|
| `elo` | Rating, starts at 1505 |
| `gp`, `wins`, `net` | Season-to-date games, wins, cumulative point differential |
| `last10` | Deque of the last 10 (win, margin) pairs |
| `last_date` | Date of the previous game, used for rest |
| `prev_w_pct`, `prev_net` | Last season's final values, used as priors |

**Season rollover** (`start_season`): last season's record becomes the prior, and Elo regresses
25% toward the mean: `elo = 0.75·elo + 0.25·1505`. Rosters change, so ratings shouldn't carry
over fully.

### 2.3 Elo (FiveThirtyEight-style)

```python
ELO_BASE, ELO_K, ELO_HCA, ELO_CARRY = 1505.0, 20.0, 100.0, 0.75

p_home = 1 / (1 + 10 ** (-(elo_home + HCA - elo_away) / 400))
mov_mult = (|margin| + 3) ** 0.8 / (7.5 + 0.006 * winner_elo_diff)   # blowouts count more,
delta = K * mov_mult * (home_won - p_home)                            # expected blowouts less
elo_home += delta;  elo_away -= delta                                 # zero-sum
```

`HCA = 100` Elo points is about a 64% home win chance between equal teams. That is
slightly above the modern ~55–58%, and the model learns the correction from the data.

**Neutral sites:** for games with `neutral = true`, both the probability and the update use
`hca=0`. Otherwise a team that "wins at home" in London would earn too little Elo, and a loss
would cost too much.

### 2.4 Early-season shrinkage

After 2 games a team's win% is 0% or 100%, which is noise. Season-to-date stats are shrunk
toward last season with a weight of `PRIOR_GAMES = 10` pseudo-games:

```python
w_pct = (wins + 10 * prev_w_pct) / (gp + 10)
net   = (net_total + 10 * prev_net) / (gp + 10)
```

By mid-season the prior barely matters. On opening night it is the only signal.

### 2.5 Feature list (`GAME_FEATURES`, 26 columns)

Per team, prefixed `home_` / `away_`:

| Feature | Description |
|---|---|
| `elo` | Current rating |
| `w_pct`, `net` | Shrunk season-to-date win% and point differential per game |
| `l10_w_pct`, `l10_net` | Last-10 form (falls back to the prior before game 1) |
| `prev_w_pct`, `prev_net` | Last season's final values |
| `gp` | Games played this season, so the model knows how much to trust `w_pct` |
| `rest`, `b2b` | Days since last game (capped at 7), back-to-back flag |

Matchup-level:

| Feature | Description |
|---|---|
| `elo_diff`, `net_diff`, `w_pct_diff` | Home minus away |
| `elo_prob` | Elo's own win probability (no home court on neutral sites), which doubles as the baseline |
| `is_playoff` | Playoff games behave differently (no tanking, best-vs-best) |
| `is_neutral` | No home crowd |

### 2.6 End-of-season snapshots

`build_game_features` also returns `snapshots[(season, team_abr)]`: each team's state after
its last game, with `rest=2, b2b=0`. These power hypothetical matchups in the app ("2016
Warriors at home vs. 1996 Bulls"), which have no real game row to draw features from.

---

## Step 3 — Training protocol (`train()`)

```
1980          1981 ─────────────────────── 2024   2025 ── 2026
burn-in       train (TimeSeriesSplit CV inside)   held-out test (always the last 2 seasons)
(dropped)
```

1. **Burn-in.** 1980 is dropped because everyone's Elo is 1505 and there is no previous
   season.
2. **Temporal holdout.** The last `TEST_SEASONS = 2` seasons are never touched during
   development.
3. **CV.** `TimeSeriesSplit(n_splits=5)` on the training games, which are already in date
   order. Each fold trains on the past and validates on the future. Each fold prints
   XGBoost, Elo-only and always-home accuracy side by side.
4. **Model.** Same hyperparameters as v1:
   ```python
   XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss", random_state=42)
   ```
   There is no `StandardScaler`, because trees don't need one.
5. **Calibration.** `CalibratedClassifierCV(xgb, cv=TimeSeriesSplit(5), method="isotonic")`.
   The calibration folds are time-ordered too, so calibration never peeks at the future.
6. **Held-out report.** Four rows: calibrated XGB, raw XGB, Elo-only, always-home.
7. **Final refit** on all seasons, saved with the held-out metrics. Those are the numbers
   the app shows.

### Metrics, and why four of them

| Metric | Answers | Better |
|---|---|---|
| Accuracy | How often the favourite wins | ↑ |
| ROC-AUC | Does it rank games correctly by win chance? | ↑ |
| Log-loss | Are the probabilities sharp *and* right? Punishes confident misses | ↓ |
| Brier | Mean squared error of the probability | ↓ |

For a probability product, log-loss and Brier matter more than accuracy. A 51% and a 95%
prediction both count as "correct" for accuracy.

### Results (held out: 2024-25 and 2025-26, 2,641 games)

| model | accuracy | AUC | log-loss | Brier |
|---|---|---|---|---|
| XGBoost + isotonic (shipped) | 0.664 | 0.726 | 0.614 | 0.212 |
| XGBoost raw | **0.670** | **0.730** | **0.610** | **0.210** |
| Elo only | 0.661 | 0.724 | 0.619 | 0.215 |
| always home | 0.550 | 0.500 | 0.695 | 0.251 |

**How to read this:**
- Pre-game NBA models usually land at 64–68% accuracy, because basketball has real
  randomness. If you ever see 75%+, look for leakage before celebrating.
- XGBoost beats Elo by a small but consistent margin.
- On this holdout, isotonic calibration slightly **hurts** compared with the raw model. Each
  calibration fold of `TimeSeriesSplit` trains its XGBoost on less (and older) data, and
  the raw model is already well calibrated with `logloss` as its objective. Whether to ship
  the raw model is an open item in `NEXT_STEPS.md`.
- The previous run, with 2023-24 and 2024-25 held out, gave 0.651 / 0.617 for the calibrated
  model, so results move a little with the test window.

---

## Step 4 — The saved bundle (`models/game_predictor.pkl`)

```python
{
  "model":        CalibratedClassifierCV,      # final, fit on all seasons
  "feature_cols": GAME_FEATURES,
  "snapshots":    {(season, team_abr): {feature: value}},
  "cv":           DataFrame,                   # per-fold CV metrics
  "test_report":  DataFrame,                   # the 4-row table above
  "test_seasons": (2025, 2026),
  "accuracy", "auc", "log_loss", "brier":  float   # calibrated XGB on the held-out seasons
}
```

---

## Step 5 — Inference API

```python
from ml.game_predictor import predict_winner, available_seasons, available_teams, \
                              get_model_stats, feature_importance

predict_winner("OKC", "BOS", season=2026, playoff=False, neutral=False)
# {'home_team': 'OKC', 'away_team': 'BOS', 'season': 2026,
#  'home_win_prob': 0.69, 'away_win_prob': 0.31, 'predicted_winner': 'OKC',
#  'confidence': 'high', 'elo_home_prob': 0.77, 'model_accuracy': 0.664}
```

- Teams are the **BRef abbreviation of that season**, e.g. `SEA` in 2005 and `OKC` in 2015.
  `available_teams(season)` lists the valid ones.
- `confidence` is `high` if |p − 0.5| > 0.15, `moderate` if > 0.07, otherwise `low`.
- `feature_importance()` averages XGBoost gain importance across the calibration folds.
- `neutral=True` removes home court from the Elo inputs and sets `is_neutral`. **Limitation:**
  only 207 of ~56k training games are neutral, so XGBoost barely uses `is_neutral` (OKC–BOS
  moves from 0.686 to 0.680). The Elo-only number reacts properly.

---

## Step 6 — Run & verify

```bash
export PYTHONPATH=.
python -m etl.validate               # must pass first: garbage in, garbage out
python -m ml.game_predictor train    # ~1–2 min
pytest tests/test_features.py -q
```

Checks before trusting the model:
- [ ] Every CV fold shows XGB accuracy ≥ always-home accuracy.
- [ ] Held-out XGB log-loss is below Elo-only log-loss. If it isn't, Elo alone is the better
      product.
- [ ] `elo_prob` and the `*_diff` features sit at the top of `feature_importance()`. If `gp`
      or `rest` dominate, something is wrong.

### The leakage test

`test_features_use_only_past_games` flips one game's result and asserts:
- that game's own features are **identical** (no leakage), and
- the *next* game's Elo **changes** (the state really updates).

Keep this test green whenever you add a feature.

---

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| Accuracy suddenly ~75%+ | A new feature uses post-game info | Add it inside `TeamState.features()`, which only sees prior games, and re-run the leakage test |
| `ValueError: no data for ['SEA'] in 2015` | Abbreviations change over time | Use `available_teams(season)` |
| App hangs when the LSTM page loads after a prediction | xgboost and torch ship separate OpenMP runtimes on macOS | Already handled: `torch.set_num_threads(1)` in `player_lstm._load()` |
| Metrics look stale after re-running the ETL | The bundle is a snapshot | Retrain; the app reads the pickle, not the DB |

---

## What Phase 5 Expects From This

`models/game_predictor.pkl`, plus `predict_winner`, `available_seasons`, `available_teams`,
`get_model_stats` and `feature_importance`. Those functions are the only interface the
Game Predictor page uses.
