# NBA Intelligence Platform

An end-to-end ML project on 46 NBA seasons (1979-80 → 2024-25): a polite, resumable
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
python -m etl.pipeline                   # full history 1980–2025 (~1 h, resumable)
python -m etl.validate                   # data-quality gate — must pass before training

python -m ml.game_predictor train
python -m ml.similarity train            # UMAP ≈ 1 min
python -m ml.player_lstm train

./run.sh                                 # Streamlit dashboard
pytest && ruff check .
```

ETL flags: `--start/--end` (season range), `--steps per_game advanced standings schedule`,
`--force` (re-parse from the HTML cache, ignoring the checkpoint), `--refresh` (re-download).

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
| `games` | `game_id` (BRef box-score id) | team abbreviations, `game_type` ∈ regular / play_in / playoff / cup_final |
| `etl_checkpoint` | `(season, step)` | written only when a step succeeds |

## Results (first full run, 2026-10-02)

The data covers 46 seasons, 55,132 games, 3,716 players and 1,284 team-seasons, and `etl.validate` passes all
checks. The regular-season home-win rate is 60.4% (v1 had 39.4% because of inverted scores).

**Game predictor.** Held out on 2023-24 and 2024-25 (2,638 games):

| model | accuracy | AUC | log-loss | Brier |
|---|---|---|---|---|
| XGBoost + isotonic | **0.651** | **0.722** | **0.617** | **0.214** |
| Elo only | 0.644 | 0.716 | 0.626 | 0.218 |
| always home | 0.546 | 0.500 | 0.697 | 0.252 |

**Career LSTM.** Test MAE in real units on target seasons 2021–2025 (1,008 windows):

| model | pts | reb | ast | stl | blk | fg% |
|---|---|---|---|---|---|---|
| LSTM | 2.17 | 0.78 | 0.63 | 0.177 | 0.151 | 0.031 |
| Ridge on the same window | **2.14** | **0.76** | **0.61** | **0.169** | **0.139** | **0.030** |
| weighted avg (5/4/3/2) | 2.74 | 0.91 | 0.70 | 0.188 | 0.146 | 0.030 |
| last season | 2.32 | 0.82 | 0.66 | 0.190 | 0.157 | 0.034 |

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
  BRef marks play-in games, and the season's last "NBA Cup" game is the non-counting final.

## Layout

```
db/      models.py (schema), connection.py (get_engine, query_df), init_db.py
etl/     scraper.py (fetch + cache), parsers.py (pure HTML→DataFrame), loader.py (upserts),
         pipeline.py (CLI), validate.py (quality gate), teams.py (franchise map)
ml/      features.py (shared), game_predictor.py, similarity.py, player_lstm.py
app/     Home.py, utils.py, pages/
tests/   parser tests on tests/fixtures/*.html.gz, feature/leakage tests
```
