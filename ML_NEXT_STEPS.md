# ML Next Steps — the problems to solve

Your work plan for the machine-learning side. Each item is a **problem** with evidence, the
steps to solve it, a "done when" you can check, and the files involved. Ordered by priority.
Broader engineering items live in `NEXT_STEPS.md`; rewrite status lives in `REWRITE_PROGRESS.md`.

## Ground rules (apply to every item)

1. **Never tune on the test seasons.** The last 2 seasons (2025, 2026) are held out. Develop
   with `TimeSeriesSplit` or a validation season (e.g. 2024), and score the test seasons once,
   at the end.
2. **Always compare against the baselines.** A change that beats the current model but not
   Elo / Ridge / last-season is not a win.
3. **One change at a time, logged.** Write down every run (date, change, validation metrics)
   in a `metrics.jsonl` or a notebook table, so you can tell what actually helped.
4. **Keep the leakage test green** (`tests/test_features.py`) whenever you add a game feature.

## Step 0 — Setup (before any ML work)

```bash
git status                       # commit the pending rewrite modules 6-8 first
                                 # (commands in REWRITE_PROGRESS.md)
git switch -c ml/<problem-name>  # one branch per problem below
export PYTHONPATH=. && pytest -q # 53 passed = clean starting point
```

Current numbers to beat (held out: 2024-25 and 2025-26):

| Model | Metric | Now | Baseline |
|---|---|---|---|
| Game predictor (XGBoost) | log-loss / accuracy | **0.610 / 0.670** | Elo 0.619 / 0.661 |
| Next-season LSTM | mean normalised MAE | **0.364** | Ridge **0.350** (better!) |
| Similarity | (no metric yet) | 9 clusters, fixed k | — |

---

## Problem 1 — The game model gets neutral sites wrong *(highest priority)*

**Evidence:** OKC (68-14, the best team) vs BOS: 62% at home, but **50%** on a neutral
site. Elo says the home edge is only a few points. Only 207 of ~56k training games are
neutral, mostly near coin-flip 2020 bubble games, so the tree model learned "neutral means
50/50".

**Steps**
1. Write a test first: two teams with identical snapshots on a neutral site → p ≈ 0.50, and
   for one real matchup, home p − neutral p ≈ Elo's home edge (a few points, not 12).
2. Try **symmetric augmentation**: train on each game twice, once as is and once with the
   teams swapped (target flipped, `home_*` ↔ `away_*`), plus an explicit `home_court` input
   (+1 home, −1 away, 0 neutral). Home advantage then becomes a single learned feature.
3. Alternative, simpler: drop `is_neutral` from the model and adjust the logit afterwards
   using the home-court effect estimated on non-neutral games.
4. Compare both on validation log-loss. Neither may be worse than 0.610.

**Done when:** the step 1 test passes **and** held-out log-loss ≤ 0.610.
**Files:** `ml/features.py` (`matchup_features`, `build_game_features`),
`ml/game_predictor.py`, `tests/test_features.py`.

---

## Problem 2 — Are the game probabilities trustworthy? (calibration)

**Evidence:** isotonic calibration, fit with `TimeSeriesSplit` folds, made log-loss worse
(0.614 vs 0.610), so the shipped model is raw. Nobody has looked at a reliability curve yet:
is "70%" really 70%?

**Steps**
1. Plot a reliability diagram (10 bins: predicted vs actual home-win rate) on the validation
   season, and compute the expected calibration error (ECE).
2. If it's miscalibrated: fit isotonic **and** sigmoid (Platt) calibration on **one held-out
   training season** (not folds), and compare log-loss, Brier and ECE.
3. Ship the best of raw / isotonic / sigmoid, chosen on validation, and store which one in
   the model bundle.

**Done when:** a reliability plot is in a notebook, the choice is justified by numbers, and
ECE is reported in `test_report`.
**Files:** `ml/game_predictor.py`, `notebooks/02_model_eval.ipynb` (new).

---

## Problem 3 — The LSTM loses to a linear model

**Evidence:** test MAE in real units: LSTM pts 2.23 vs Ridge 2.19, and Ridge wins on all 6
stats (normalised 0.364 vs 0.350). With ~5.7k training windows, the sequence model adds
nothing that Ridge can't already do.

**Steps** (in this order; stop when the LSTM beats Ridge on validation)
1. **Richer inputs:** add `ts_pct, usg_pct, bpm, fg3a_rate, ast_pct, trb_pct` (already in
   `player_seasons`) to `LSTM_INPUT_COLS`. Give Ridge the same inputs too, to keep it fair.
2. **Ridge as the base:** feed Ridge's prediction to the LSTM as its starting point instead
   of last season (the network then learns Ridge's mistakes).
3. **Light tuning** on validation only: hidden size (32/64/128), dropout, and epochs with
   early stopping.
4. If it still loses: **ship Ridge** for projections, keep the LSTM as a documented
   experiment, and say so in the app.
5. Bonus: predict a range (quantile loss at 10/50/90%) instead of one number. "19–25 PPG"
   is more honest than "22.6".

**Done when:** the shipped projection model beats Ridge, last-season and weighted-average on
**validation**, then confirmed once on test, with per-stat MAE in real units reported.
**Files:** `ml/features.py` (section 4), `ml/player_lstm.py`.

---

## Problem 4 — Squeeze more out of the game predictor

**Evidence:** XGBoost beats Elo by under 1 point of accuracy. Its features are mostly Elo,
form and rest; schedule density, travel and playoff context are missing.

**Steps** (one feature at a time; keep any that improves validation log-loss)
1. Rest **difference** (home rest − away rest) and games played in the last 7 days.
2. Travel: distance or time-zone change since the last game (needs a small arena-location
   table).
3. Playoff context: game number in the series, series score (from consecutive playoff games
   between the same two teams).
4. Light hyperparameter search (`max_depth`, `min_child_weight`, `n_estimators`) inside
   `TimeSeriesSplit`.

**Done when:** held-out log-loss < 0.610 with the leakage test still passing.
**Files:** `ml/features.py` (`TeamState`, `matchup_features`), `ml/game_predictor.py`.

---

## Problem 5 — Is 9 the right number of player archetypes?

**Evidence:** `k = 9` was inherited from v1, never chosen. Labels look right for stars and
bigs, but some are debatable (Klay Thompson 2026 → "Bench Shooter").

**Steps**
1. For k = 6…14: silhouette score, plus **stability**, i.e. cluster agreement (adjusted
   Rand index) across 5 random seeds.
2. Pick the k that is stable *and* interpretable, then check that the template names still
   fit the centroids.
3. Spot-check 10 well-known players per archetype.

**Done when:** k is chosen with a plot and a written reason, and the labels survive the
spot-check.
**Files:** `ml/similarity.py`, `notebooks/` (new evaluation notebook).

---

## Problem 6 — Track experiments (do this alongside Problem 1)

**Steps:** every `train()` appends one line to `models/metrics.jsonl` with the timestamp, git
commit, model name, data row count and the held-out metrics. Add a small notebook that plots
the history.

**Done when:** you can answer "did change X help?" from the log alone.
**Files:** `ml/game_predictor.py`, `ml/similarity.py`, `ml/player_lstm.py`.

---

## Suggested order

1 (neutral sites) + 6 (metrics log) → 2 (calibration) → 3 (LSTM) → 4 (features) → 5 (archetypes).

Rewrite modules 9–10 (tidying `similarity.py` and `player_lstm.py`) are still pending.
Either ask Claude to finish them before you start Problem 3/5, or fold that tidy into your
own work on those files.
