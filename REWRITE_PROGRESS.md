# Rewrite Progress — from "good" to "production level"

Status log for the two-axis plan agreed on 2026-10-05. **Read this first when coming back.**

- **Axis 1 — project phases:** ETL → game model → similarity → LSTM → app (→ optional AI layer).
- **Axis 2 — engineering maturity:** L0 Minimal → L1 Engineered → L2 Resilient →
  L3 ML-in-prod → L4 Operated.
- **Order:** skeleton first. Every phase is rewritten at L0, then the whole system moves up
  one level per iteration.
- **How we work:** Claude rewrites one module at a time and verifies it (tests + parity with
  the reference). Claude then explains it with self-check questions, you read and run it,
  and **you commit**. Claude never pushes.

---

## Where we are (last updated 2026-10-09)

- **Git:** branch `rewrite/minimal`. The pre-rewrite code is tagged `v0-reference` (commit
  `2f419c1`, which is also `main`).
- **Databases:**
  - `nba_platform_v2` is the **reference**, with full data 1980–2026. Leave it untouched; it's
    what the rewrite is compared against.
  - `nba_platform_l0` is the rewrite's DB, with all 47 seasons loaded by the rewritten
    pipeline (module 5) and **identical** to the reference.
  - `nba_platform_test` is wiped by `tests/test_loader.py` on every run.
  - `nba_platform` is the old v1 data. Don't use it.
- **`.env`:** `DATABASE_URL` now points at `nba_platform_l0` (switched at module 5). The
  reference URL is kept as a comment line above it.
- **Models (`models/*.pkl`)** were trained on `nba_platform_v2` data by the old code. That's
  fine until modules 7–10 retrain them.

### L0 module checklist

| # | Module | Status |
|---|---|---|
| 0 | Setup: commit, tag `v0-reference`, branch, create `nba_platform_l0` | ✅ done |
| 1 | `db/models.py`, `db/init_db.py` | ✅ committed (`9fc346f`) |
| 2 | `etl/parsers.py` (280 → 235 lines) | ✅ verified, **⚠️ NOT COMMITTED YET** |
| 3 | `etl/scraper.py` (merged `get_players`) | ✅ verified, **⚠️ NOT COMMITTED YET** |
| 4 | `etl/loader.py` (replace-season pattern; column lists removed from models) | ✅ verified, **⚠️ NOT COMMITTED YET** |
| 5 | `etl/pipeline.py` (3 steps) + first full load into `nba_platform_l0` | ✅ verified, **⚠️ NOT COMMITTED YET** |
| 6 | `etl/validate.py` (declarative `CHECKS = {name: SQL returning bad rows}`) | ⏭ **next** |
| 7 | `ml/features.py` (tidy; guarantee the `neutral` column) | ⏳ |
| 8 | `ml/game_predictor.py` (ship raw XGBoost, drop calibration) | ⏳ |
| 9 | `ml/similarity.py` (light tidy) | ⏳ |
| 10 | `ml/player_lstm.py` (keep LSTM, light tidy) | ⏳ |
| 11 | `app/` (adapt to API changes) | ⏳ |
| 12 | Tests + docs: update `PHASE1–5` guides, write `ENGINEERING_LEVELS.md` | ⏳ |

### Module 1 — what was done (2026-10-05)

- `db/models.py`: all 43 stat columns are declared explicitly (no more `setattr` loop), with
  comments and a module docstring stating the 3 conventions (season = ending year,
  abbreviations as of that season, `bref_id` identity). `PER_GAME_COLS`/`ADVANCED_COLS` are
  kept temporarily for the current loader and get removed in module 4.
- `db/init_db.py`: the hand-rolled `MIGRATIONS` list is removed (Alembic arrives in L1). New
  `init_db(engine, drop=False)` function used by both the CLI and the tests. Views are a
  `{name: query}` dict, and everything runs in one transaction.
- `tests/test_loader.py`: the fixture now calls `init_db(engine, drop=True)`.
- **Verified:** the schema built twice into `nba_platform_l0` (idempotent). A `pg_dump
  --schema-only` diff against `nba_platform_v2` is identical apart from `games.neutral`'s
  column position, a harmless leftover of v2's `ALTER TABLE`. 46 tests pass, ruff is clean.
- **Self-check questions** (answer them before moving on):
  1. If `pts` changed from `Float` to `Integer` and `create_all()` ran again, what happens to
     the existing table? Why does that lead to Alembic?
  2. Why must the views be dropped before `drop_all()` and created after `create_all()`?
  3. What breaks in the `franchises` view without the `teams.` prefix on `last_season`?

### Modules 2–5 — what was done (2026-10-09)

**Module 2, `etl/parsers.py`.** Every fallback was measured on all 47 cached seasons first:
- Removed, because they fired **0 times**: the HTML-comment table search, the old-layout
  aliases, the `confs_standings` fallback, the box-score/date-text game-id fallback,
  `drop_duplicates`, and the unused `table_ids`/`_first` helpers.
- Kept, because they fire: skipping rows without a player id (**94** "League Average" rows).
  Skipping unplayed games is also kept, since BRef lists future games with blank scores.
- Helpers renamed for clarity: `read_row`, `to_float`, `team_from_link`.
- **Verified:** output identical to the `v0-reference` parser on all **588 cached pages**.

**Module 3, `etl/scraper.py`.**
- `get_players` returns per-game and advanced stats as one wide frame, via the pure
  `parsers.merge_player_tables`: an outer join with `validate="one_to_one"`. Both pages were
  checked to list identical rows, with identical name/age/pos/primary/team_list, in every
  season.
- Renames: `get_schedule` (was `get_season_schedule`) and `fetch_required` (was `_require`).

**Module 4, `etl/loader.py`.**
- **Replace-season pattern:** one transaction that runs `DELETE … WHERE season = ?`, then
  `DataFrame.to_sql`. This removes `_records`, `_upsert`, the stale-row deletes and the
  two-step partial rows.
- Players are still upserted by `bref_id`.
- Functions renamed `load_players`, `load_standings` and `load_games`.
- `PER_GAME_COLS`/`ADVANCED_COLS` are removed from `db/models.py`.

**Module 5, `etl/pipeline.py`.**
- `STEPS = {name: (extract, load)}` dispatch table with 3 steps: players → standings →
  schedule.
- Same flags and checkpoint behaviour. A 429 stops everything with exit code 2.

**Verification:**
- Full load of 47 seasons into `nba_platform_l0` from the cache: 90 s, 0 failures, no network.
- Per-table md5 over natural keys is **identical** to `nba_platform_v2` for players
  (3,819), player_seasons (25,349), team_seasons (1,314), games (56,454) and both views.
- `etl.validate` passes all checks.
- Re-running 2026 changed nothing (idempotent).
- 46 tests pass, ruff is clean.

**Self-check questions:**
1. The loader deletes a season, then inserts it. What guarantees we never end up with a
   season that is deleted but not re-inserted, for example if the insert crashes?
2. Why are players upserted instead of following the replace-season pattern like the other
   tables?
3. The merge uses `how="outer"` plus `validate="one_to_one"` even though both pages always
   agree. What would an inner join silently do if BRef ever listed a player on only one
   page?
4. The parser rewrite was verified by comparing against the old parser rather than only by
   unit tests. What does that "differential testing" catch that the fixture tests can't?

**Not committed yet.** Suggested commits:
```bash
git add etl/parsers.py && git commit -m "L0 module 2: evidence-driven parser simplification (parity on 588 pages)"
git add -A etl db tests REWRITE_PROGRESS.md && git commit -m "L0 modules 3-5: merged players step, replace-season loader, 3-step pipeline"
```
`.claude/settings.json` (enables a Claude Code plugin) is untracked. Commit it if you want the
plugin setting shared, or add `.claude/` to `.gitignore`.

---

## How to resume

```bash
cd ~/Projects/personal_projects/basketball
git switch rewrite/minimal && git status          # modules 2-5 may still be uncommitted (see above)

conda activate nba                                # or /opt/miniconda3/envs/nba/bin/python
export PYTHONPATH=.
pytest -q && ruff check .                         # expect 46 passed, all checks passed
```

Then ask Claude: **"continue the rewrite with module 6 (etl/validate.py)"**.

---

## L0 verification target (end of module 12)

1. `python -m etl.pipeline --force` into `nba_platform_l0` from the HTML cache (no network,
   ~1.5 min).
2. Per-table ordered md5 over natural keys and values matches `nba_platform_v2`.
3. `python -m etl.validate` passes, `pytest` passes, `ruff` is clean.
4. Retrained models: game raw XGB ≈ 0.670 accuracy / 0.610 log-loss, similarity labels
   unchanged for known players, LSTM ≈ 0.364 normalised MAE.
5. Headless `AppTest` on all 6 pages.

---

## The maturity ladder (after L0)

**L1 — Engineered** *(a codebase a team can work on safely)*
- `pyproject.toml` packaging with console scripts instead of `PYTHONPATH=.`.
- pydantic-settings for config.
- `logging` instead of `print`.
- mypy, plus pre-commit hooks.
- GitHub Actions CI (lint, types, unit tests, and integration tests on a Postgres service
  container).
- Alembic migrations.
- docker-compose for Postgres, plus a Makefile.
- pytest markers and coverage.
- App smoke tests in pytest.

**L2 — Resilient** *(survives the real world)*
- HTTP timeouts, retries with backoff and jitter, `Retry-After`, and a token-bucket rate
  limiter.
- pandera data contracts at the parse boundary.
- Upserts and incremental in-season updates.
- An `etl_runs` metadata table and structured JSON logs.
- The validation gate becomes part of the pipeline.
- Scheduled runs with alerting.
- Raw-page lineage (`fetched_at`, sha256).
- Failure-injection tests.

**L3 — ML in production** *(trustworthy, reproducible, rollback-able models)*
- MLflow tracking and a model registry (models versioned with data snapshot and git SHA).
- Reproducible training.
- CI evaluation gates (must beat Elo / Ridge / last-season).
- Proper calibration.
- Training/serving parity tests and model cards.
- NEXT_STEPS model items (neutral sites, LSTM, predictor features, similarity stability).
- Prediction logging and drift monitoring.

**L4 — Operated service** *(run it like a product; core FDE skills)*
- FastAPI prediction service; Streamlit becomes a client of it.
- Docker images and a cloud deploy with hosted Postgres.
- Health checks, metrics and tracing (OpenTelemetry).
- Secrets management, API auth and rate limiting.
- Load test with an SLO.
- Runbook and an incident drill.
- Dependency scanning.

**Optional Phase 6 — AI layer** *(after L1)*
- A text-to-SQL agent over the DB, with read-only access, guardrails and an eval suite.

The full plan text is in `~/.claude/plans/before-moving-on-to-groovy-shell.md`. Open
feature and model work, mapped onto these levels, is in `NEXT_STEPS.md`.
