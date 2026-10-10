# Engineering Levels — from "works" to "production"

The second axis of this project. **L0 (minimal, clear, verified) is complete**: see
`REWRITE_PROGRESS.md` for how it was done. Each level below applies to the *whole* system
(ETL, models, app) and is done on feature branches, merged into `main` one at a time.
Each level ends with a short "what I learned" note added here.

| Level | Goal | Status |
|---|---|---|
| L0 Minimal | Clear code you fully understand, verified against the reference | ✅ done |
| L1 Engineered | A codebase a team can work on safely | ⏳ |
| L2 Resilient | A pipeline that survives the real world | ⏳ |
| L3 ML in production | Models you can trust, reproduce and roll back | ⏳ |
| L4 Operated | Run it like a product (core FDE skills) | ⏳ |

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
