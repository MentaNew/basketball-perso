# NBA Intelligence Platform

An end-to-end ML project on every NBA season from 1979-80 through the last completed one
(currently 2025-26, 47 seasons): a polite, resumable
Basketball-Reference scraper feeding PostgreSQL, three models, and a Streamlit dashboard.

| Layer | What |
|---|---|
| Data engineering | `etl/`: cached HTML scraping → idempotent upserts → validation gate |
| Classical ML | `ml/game_predictor.py`: XGBoost + isotonic calibration on leak-free pre-game features, compared against Elo and always-home baselines |
| Unsupervised ML | `ml/similarity.py`: per-season z-scores → PCA(10) → KMeans(9), UMAP(2) for the map; archetype names matched to centroids |
| Deep learning | `ml/player_lstm.py`: 2-layer LSTM over 4-season windows → next season, compared against ridge, weighted-average and last-season baselines |
| Product | `app/`: Home, Player Profile, Team History, Leaders, Game Predictor, Archetype Explorer |

This is a rebuild of an earlier version; `PROJECT_BLUEPRINT.md` documents the original design
and the bugs fixed here (see *Fixes vs. v1* below).

## Setup

```bash
conda activate nba                       # Python 3.11
pip install -r requirements.txt
brew services start postgresql@17
createdb -O nba_user nba_platform_v2     # or any DB; set DATABASE_URL in .env
cp .env.example .env                     # edit DATABASE_URL
```

## Run

```bash
export PYTHONPATH=.
python -m db.init_db                     # create tables (idempotent)

python -m etl.pipeline --season 2024     # smoke test one season
python -m etl.pipeline                   # 1980 → last completed season (~1 h, resumable)
python -m etl.validate                   # data-quality gate — must pass before training

python -m ml.game_predictor train
python -m ml.similarity train            # UMAP ≈ 1 min
python -m ml.player_lstm train

./run.sh                                 # Streamlit dashboard
pytest && ruff check .
```

ETL flags: `--start/--end` (season range), `--steps per_game advanced standings schedule`,
`--force` (re-parse from the HTML cache, ignoring the checkpoint), `--refresh` (re-download).
The season range ends at the last completed season: season `Y` counts as complete from
July 1 of `Y`. New seasons are picked up automatically by a plain `python -m etl.pipeline`.
After changing the schema, re-run `python -m db.init_db` (it applies the small migrations in
`MIGRATIONS`).

## Data

`season` is the ending year (`2024` = 2023-24). Every fetched page is cached in
`data/raw/{season}/{page}.html`, so parsing changes never need a re-scrape. Basketball-Reference
bans clients that exceed ~20 requests/min. The scraper sleeps 4–5 s per request and stops
on a 429.

| Table | Key | Notes |
|---|---|---|
| `players` | `bref_id` | BRef player id (no more merging namesakes); `search_name` is accent-free |
| `player_seasons` | `(player_id, season, team_abr)` | wide: per-game + advanced. Traded players have per-team rows plus `team_abr='TOT'`; `is_primary` marks the one row per player-season |
| `team_seasons` | `(team_abr, season)` | standings for both conferences in every era, `franchise_id` links relocations (SEA→OKC, NJN→BRK…) |
| `games` | `game_id` (BRef box-score id) | team abbreviations, `game_type` ∈ regular / play_in / playoff / cup_final, `neutral` (games abroad and the 2020 bubble) |
| `etl_checkpoint` | `(season, step)` | written only when a step succeeds |
| `teams`, `franchises` *(views)* | `team_abr` / `franchise_id` | abbreviation → franchise lookup with name and season span, derived from `team_seasons` |

## Results (run of 2026-10-03, data through 2025-26)

The data covers 47 seasons, 56,454 games, 3,819 players and 1,314 team-seasons, and
`etl.validate` passes all checks. The home-win rate in regular-season, non-neutral games is
60.3% (v1 had 39.4% because of inverted scores).

**Game predictor.** Held out on 2024-25 and 2025-26 (2,641 games):

| model | accuracy | AUC | log-loss | Brier |
|---|---|---|---|---|
| XGBoost + isotonic (shipped) | 0.664 | 0.726 | 0.614 | 0.212 |
| XGBoost raw | **0.670** | **0.730** | **0.610** | **0.210** |
| Elo only | 0.661 | 0.724 | 0.619 | 0.215 |
| always home | 0.550 | 0.500 | 0.695 | 0.251 |

On this holdout, isotonic calibration slightly *hurts* compared with the raw model (see
`NEXT_STEPS.md`).

**Career LSTM.** Test MAE in real units on target seasons 2021–2026 (1,221 windows):

| model | pts | reb | ast | stl | blk | fg% |
|---|---|---|---|---|---|---|
| LSTM | 2.23 | 0.78 | 0.64 | 0.181 | 0.152 | 0.031 |
| Ridge on the same window | **2.19** | **0.76** | **0.62** | **0.173** | **0.140** | **0.030** |
| weighted avg (2/3/4/5) | 2.75 | 0.91 | 0.71 | 0.189 | 0.150 | 0.031 |
| last season | 2.36 | 0.83 | 0.67 | 0.195 | 0.157 | 0.034 |

The LSTM beats the naive baselines but not a linear model. With about 6k training windows,
the sequence model doesn't earn its complexity yet. Worth trying: more inputs (advanced
stats, team context), predicting deltas with Ridge as a prior, or ensembling.

## Fixes vs. v1 (blueprint §7)

- **Inverted scores (§7.1):** cells are read by `data-stat` (`visitor_pts` / `home_pts`), not by
  header position. A test pins LAL 107 @ DEN 119 on 2023-10-24.
- **Leaky game features (§7.2):** features come only from games before tip-off: Elo with
  margin-of-victory and between-season regression, season-to-date win%/net rating shrunk
  toward last season, last-10 form, rest days and back-to-backs. A test asserts that changing
  a game's result never changes that game's own features.
- **Missing standings (§7.3):** each conference table is parsed separately, using division
  tables, which exist in every era. `etl.validate` cross-checks standings W/L against W/L
  recomputed from `games`.
- **`UNK` teams and name identity (§7.4):** `team_name_abbr` and `2TM/3TM` are handled, and
  players are keyed by `bref_id`.
- **Team names (§7.5):** abbreviations everywhere plus a franchise map.
- **Similarity (§7.6):** archetype names come from centroids (Hungarian matching to templates).
  There is a career-average mode, and raw stats are kept for tooltips.
- **LSTM (§7.7):** temporal split by target season, age and season-gap inputs, and baselines
  with per-stat MAE in real units.
- **Hygiene (§7.8):** pinned requirements, pytest on saved HTML fixtures, ruff, no deprecated
  `errors="ignore"`.

## Changes from the original design

- The **totals** page is no longer scraped. Per-game × games covers it, and it saves a request
  per season.
- `db/models.py` is the single schema source (`python -m db.init_db`), replacing `schema.sql`.
- Month pages are discovered from each season's schedule index, which covers lockouts and the
  2020 bubble's July–October 2020 months.
- Game types are inferred: a team's first W+L games (from standings) are its regular season,
  BRef marks play-in games, and the season's last "NBA Cup" / "In-Season Tournament" game is
  the non-counting final.
- Neutral sites: games abroad (from BRef's "at London, England"-style remarks) and the 2020
  bubble restart are flagged `neutral`. Elo then drops home court, and the predictor has an
  `is_neutral` input and a "Neutral site" toggle. Domestic and Canadian alternate venues
  (Celtics in Hartford, Raptors at Maple Leaf Gardens) stay home games: their home-win rate
  (57.8%) shows a real home edge.

## Layout

```
db/      models.py (schema), connection.py (get_engine, query_df), init_db.py
etl/     scraper.py (fetch + cache), parsers.py (pure HTML→DataFrame), loader.py (upserts),
         pipeline.py (CLI), validate.py (quality gate), teams.py (franchise map)
ml/      features.py (shared), game_predictor.py, similarity.py, player_lstm.py
app/     Home.py, utils.py, pages/
tests/   parsers, game types, scraper (faked network), loader (needs TEST_DATABASE_URL),
         feature/leakage tests
```

Guides: `PHASE1_ETL_GUIDE.md` … `PHASE5_STREAMLIT_APP_GUIDE.md` explain each layer;
`NEXT_STEPS.md` lists what's open.
