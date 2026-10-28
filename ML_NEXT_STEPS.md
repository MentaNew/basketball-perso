# ML: status, results and next ideas

Last update 2026-10-27. Every model below is a working baseline built and improved by Claude.
Each section ends with the most promising ideas for when you take over.

**Ground rules (every change):**
- Never tune on test seasons.
- Compare with the baselines.
- Every training run is logged in `models/metrics.jsonl` (`ml.experiments.read_runs()`).
- Keep `tests/test_features.py` (the leakage test) green.

**Retrain after merging** (models are git-ignored):

```bash
python -m ml.season_wins train
python -m ml.game_predictor train
python -m ml.player_lstm train
python -m ml.similarity train
```

---

## Track A — Next-season team wins · `ml/season_wins.py` · branch `ml/season-wins`

**Model:** Ridge regression on:
- last season's wins and point differential;
- **roster strength**: the players on this season's roster, valued by their previous two
  seasons (VORP and minutes-weighted BPM, weighted 2:1 Marcel-style);
- the roster's age.

| MAE in wins | Validation 2022–24 | Test 2025–26 | Backtest 2005–24 (510 predictions) |
|---|---|---|---|
| Always 41 | 9.47 | 11.10 | — |
| Last season regressed 30% | 7.86 | 9.01 | — |
| Last season only (Ridge) | 7.80 | 8.76 | 7.02 |
| **Shipped model** | **6.66** | **8.26** | **6.81** |
| Vegas preseason line | 6.59 | 7.87 | 6.56 |

- **Use `backtest()` to decide.** It covers ~500 predictions vs 90 in validation; several
  "gains" on validation were noise there.
- **Tried, no gain:** progression (Y-1 minus Y-2), lagged payroll / cap, roster continuity,
  splitting VORP by age, gradient boosting (6.73 on validation vs Ridge 6.67), centring each
  season to an average of 41 wins.
- **Known leak (small):** historical rosters come from who actually played, so players who sat
  out the whole season are invisible. That's about 1% of VORP, mostly known in October
  anyway (Durant's Achilles, Jordan retiring). The upcoming season uses today's real rosters.
- **2026-27:** `python -m ml.season_wins predict`. It correlates 0.87 with Kalshi's market
  medians. The projections sum to 1,263 wins (a season has exactly 1,230).

**Next ideas (biggest expected gain first):**
1. **Players returning from injury.** IND (33.8 vs market 45.7) is Haliburton: his missed
   2025-26 makes Marcel underrate him. Use the last *healthy* season.
2. **Player development.** WAS (21.0 vs market 36.9) has a very young roster. Project each
   player's next season with the career model (Track C) instead of last season's stats.
3. A projected-minutes model (who starts), instead of last season's minutes.
4. A page comparing projections, the Kalshi market and your own picks
   (`predictions/2026-27.xlsx`).

## Track B — Game model · `ml/features.py`, `ml/game_predictor.py` · branch `ml/game-model`

- **Elo:** home advantage is now estimated each season from the previous 5 seasons'
  home-win rate (`EloConfig(hca_window=5)`). Home advantage shrank from ~65% home wins in
  the 1980s to ~55% today. In Elo points it's ≈36 for 2026, not the borrowed constant 100.
- **Model:** symmetric logistic regression. XGBoost overfit (time-ordered CV 0.6028, worse
  than Elo alone at 0.5989; logistic 0.5975). Every game is trained on both as played and
  mirrored, and predictions average the two. So home advantage lives only in `home_court`,
  and a neutral-site game between equal teams is exactly 50%.

| Held-out 2024-25 + 2025-26 | Accuracy | Log-loss | ECE |
|---|---|---|---|
| Original XGBoost | 0.670 | 0.6099 | — |
| Elo only (tuned) | 0.670 | 0.6062 | 0.016 |
| **Shipped logistic** | **0.677** | **0.6029** | 0.028 |
| Playoffs only: logistic / Elo | 0.627 / 0.586 | 0.650 / 0.650 | |

- **Fixed:** neutral sites. OKC–BOS is 67% at home and 60% neutral, where it used to be 50%.
  The app warning is gone.
- **Calibration (B4):** ECE 0.028, i.e. probabilities within ~3 points. No calibration step needed.
- **Grid-searched K / carry-over:** the defaults (20 / 0.75) were already best.

**Next ideas:**
1. **Playoff mode:** series state (game number, series score). Needs more test playoff games
   to validate (169 in the test seasons).
2. Rest difference, games in the last 7 days, travel.
3. A playoff-series simulator (Monte-Carlo best-of-7) for the app.

## Track C — Player models · `ml/player_lstm.py`, `ml/similarity.py` · branch `ml/player-models`

**Projections:**
- Inputs: TS%, usage and BPM were added. The LSTM is now 64×1 (it was 128×2, which overfit).
- The model served is chosen on validation. The **LSTM + Ridge average (blend)** won, and it
  is also best on test:

| Test normalised MAE | 2021–26 |
|---|---|
| LSTM, before | 0.364 |
| Ridge | 0.351 |
| LSTM, now | 0.357 |
| **Blend (served)** | **0.350** |

`predict_next_season()` now returns which model produced the projection.

**Archetypes:**
- k = 9 was unstable across random seeds (agreement 0.73). **k = 8** is stable (0.97).
- Two templates were added after inspecting the centroids: "Inefficient Guard" and
  "Low-Usage Big". Known players check out (Curry and Jokić are Franchise Stars, Gobert a
  Traditional Big, Klay now a 3-and-D Wing).
- The silhouette is low (~0.14): player styles are a continuum, not crisp groups.

**Next ideas:**
1. Quantile projections (10 / 50 / 90%) instead of one number.
2. A "comps" sanity set (20 known similar-player pairs) to evaluate similarity.
3. Feed projected player stats into Track A (see idea A2).

## Not done (needs your call)
- **Side projects** (`SIDE_PROJECTS.md`): daily Kalshi snapshots should start now, while
  the season is young.
- **Engineering levels L1+** (`ENGINEERING_LEVELS.md`).
