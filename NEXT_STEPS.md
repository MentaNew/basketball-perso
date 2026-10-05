# Next Steps

Status as of 2026-10-03: Phases 1–5 are built and verified. 47 seasons (1980–2026) are
loaded, `etl.validate` passes, all three models are trained, the six app pages render,
46 tests pass and ruff is clean.

Below is everything still open, in priority order. Each item names the files to touch and
what "done" means. Phase guides: `PHASE1_ETL_GUIDE.md` … `PHASE5_STREAMLIT_APP_GUIDE.md`.

---

## Done (2026-10-03)

- **2025-26 season loaded.** `etl/pipeline.py` now ends at `last_completed_season()`, which
  counts season `Y` from July 1 of `Y`, so future seasons are picked up automatically. 2026
  validated with 1,230 regular-season games and its Cup final detected.
- **`PHASE1_ETL_GUIDE.md` rewritten** for the rebuilt pipeline.
- **Neutral sites.** `games.neutral` covers 35 games abroad and 172 in the 2020 bubble. Domestic
  and Canadian alternate venues were checked and stay home games (57.8% home-win rate).
  Elo drops home court on neutral games, the predictor has `is_neutral` and a UI toggle, and
  `etl.validate` checks the flag.
- **Wasted request removed.** A season's first month page is no longer fetched twice
  (verified identical to the index page for all 46 cached seasons).
- **Phase 1 gaps closed (2026-10-05).** `teams` / `franchises` lookup views (blueprint §7.5);
  pure, tested `infer_game_types`; loads now delete rows that vanished from the source;
  player-name corrections propagate (a `WHERE` clause used to block them); loader
  integration tests (§9) and scraper tests; `etl.validate` checks completeness, box totals
  and franchises.
- **Player picker bug.** Namesakes with identical career spans (the two 2013-14 Tony
  Mitchells) had identical labels. Labels now add the teams.

---

## P0 — Findings from the latest retrain

### 1. Ship the raw or the calibrated game model?
On the new holdout (2024-25 and 2025-26), isotonic calibration made things slightly
**worse**: log-loss 0.614 vs 0.610 raw, accuracy 0.664 vs 0.670. Each `TimeSeriesSplit`
calibration fold trains its XGBoost on less and older data.
- **Do:** in `ml/game_predictor.py`, choose between raw, isotonic and sigmoid by log-loss on
  the **last training season** (never the test seasons). Alternatively, calibrate a model fit
  on all training data using a held-out final training season (`cv="prefit"`-style).
- **Done when:** the shipped model is the best of the three on validation, and the choice is
  stored in the bundle and shown in the app's "How good is the model?" expander.

### 2. Make the predictor respect neutral sites
Only 207 of ~56k games are neutral, so XGBoost barely splits on `is_neutral`. OKC–BOS moves
from 0.686 to only 0.680 when the toggle is on, while the Elo-only probability reacts
properly.
- **Do (pick one):** (a) monotone or interaction constraints, (b) predict with the model and
  then shift by the learned home-court effect in logit space when neutral, or (c) train on
  the symmetric augmentation (each game also seen with teams swapped and `is_home` = ±1, so
  home court becomes a single explicit feature).
- **Done when:** a neutral-site prediction for two teams with equal snapshots is ≈ 0.50.

---

## P1 — Models

### 3. Make the LSTM earn its place (or ship Ridge)
It still loses to Ridge on every stat after the retrain (mean normalised MAE 0.364 vs 0.350,
Phase 4 guide).
Options, cheapest first:
1. **Model selection on validation:** in `ml/player_lstm.py`, train both and keep whichever wins
   on the validation seasons for `predict_next_season`. The UI says which model produced the
   projection.
2. **Richer inputs:** add `ts_pct, usg_pct, bpm, fg3a_rate, ast_pct, trb_pct` (already in
   `player_seasons`) to `LSTM_INPUT_COLS`, plus team context (team `net` from Phase 2 snapshots).
3. **Ridge prior + LSTM residual:** feed the Ridge prediction as an input or residual base
   instead of last season.
4. **Uncertainty:** quantile heads (10/50/90) and a range in the UI ("22.6 PPG, 80% range
   19–25"). That is more useful than a point estimate.
- **Done when:** whichever ships beats Ridge on test, or Ridge is the shipped model and the
  LSTM is documented as the experiment.

### 4. Game predictor improvements
- **SHAP explanations** on the predictor page ("Elo gap +6%, OKC on a back-to-back −2%").
  Use `xgboost`'s `pred_contribs=True` to avoid a new dependency.
- **Better features:** a rest differential, games in the last 7 days, travel/time zones, and
  a playoff series state (game number, series score).
- **Light tuning:** search `max_depth`, `min_child_weight` and `n_estimators` inside the
  existing `TimeSeriesSplit`. Never on the test seasons.
- **Done when:** held-out log-loss improves over 0.610 with the leakage test still green.

### 5. Similarity robustness
- Pick `k` with silhouette and seed stability (cluster agreement across 5 seeds via ARI)
  instead of a fixed 9. Report it in `train()`.
- Spot-check the labels after any change. Klay Thompson's 2026 season currently reads
  "Bench Shooter", which fits an aging shooter but is worth a look.

---

## P2 — Engineering

### 6. CI
`.github/workflows/ci.yml` running `ruff check .` and `pytest` on Python 3.11. The current tests
use only the saved HTML fixtures and toy data, so **no database is needed in CI**.

### 7. App integration test
The loader integration tests are done (`tests/test_loader.py`, against `TEST_DATABASE_URL`).
Still open: move the headless `AppTest` checks (Phase 5 guide, Step 3) into
`tests/test_app.py`, marked `@pytest.mark.integration` because they need the real DB and
trained models.

### 8. Model metrics log
Every `train()` appends to `models/metrics.jsonl`: timestamp, git SHA, data row counts and
held-out metrics. That gives a history for spotting regressions, without needing MLflow.

## P3 — Product & extensions

### 9. In-season updates (2026-27 tips off in late October 2026)
- `python -m etl.pipeline --current`: refresh only the latest month page(s), standings and
  per-game/advanced for the in-progress season, using `--refresh` for those pages only.
- Game-type inference still works mid-season, because partial standings have W+L equal to
  games played.
- The predictor then becomes useful for real upcoming games, which needs the scheduled-but-
  unplayed rows (currently skipped by `parse_schedule`).

### 10. Playoff series simulator
Monte-Carlo a best-of-7 with `predict_winner(..., playoff=True)`, alternating home court
2-2-1-1-1. Show the series-win probability and the distribution over 4–7 games. It works for
cross-era matchups too.

### 11. Notebooks
- `notebooks/01_eda.ipynb`: era trends and data-quality plots (from `etl.validate`'s queries).
- `notebooks/02_model_eval.ipynb`: calibration curves, per-season accuracy, LSTM error by age.

### 12. Deployment
- `docker-compose.yml` with Postgres plus the app, for local reproduction.
- Hosted: Streamlit Community Cloud plus Neon/Supabase Postgres (`pg_dump` the ~47 MB DB).
- Model artifacts are git-ignored. Publish them as a GitHub release asset, or train in CI and
  upload them. `data/raw/` (~320 MB) does not need to be deployed.

### 13. More data
Playoff box scores, player game logs (in-season form features) and shot locations (2000+).
Each needs new parsers plus fixtures in the existing pattern.
