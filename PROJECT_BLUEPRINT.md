# NBA Intelligence Platform — Project Blueprint

> A self-contained spec for rebuilding this project from scratch. It covers the goal, the architecture,
> every design decision found in the original code (with exact hyperparameters), the bugs that were
> verified in the existing data and models, and what is still left to build.
>
> Compiled 2026-09-30 from the original project folder (`ML_DL_projects/BasketBall`).

---

## 0. TL;DR

- **What it is:** a personal end-to-end ML project. It scrapes 46 NBA seasons (1979-80 → 2024-25) from
  Basketball-Reference into PostgreSQL, trains three models on top of the data, and serves them in a
  dark-themed Streamlit dashboard.
- **The three models:**
  1. **Game predictor:** XGBoost win probability for a home vs. away matchup.
  2. **Player similarity and archetypes:** era-normalised stats → PCA → UMAP → KMeans → "who plays like X?"
  3. **Career-arc LSTM:** predicts a player's next-season per-game stats from their last 4 seasons.
- **Where it stopped:** the ETL ran, the database is populated, and all three models were trained
  (artifacts exist in `models/`). The Streamlit pages were never finished: only a helpers module
  existed. Most `.py` source files were later deleted, so the ML code below was **recovered from
  leftover `__pycache__/*.pyc` bytecode**.
- **Critical:** the existing data has **3 serious bugs** (§7). The game results are inverted, every
  player's team is `UNK`, and the West standings are missing. The trained game model is therefore
  invalid. Fix these before anything else.

---

## 1. Goal & Essence

Build a portfolio-grade "NBA analytics platform" that shows the full ML lifecycle on real data:

| Layer | Purpose |
|---|---|
| **Data engineering** | A robust, idempotent, resumable scraper → relational DB (45+ seasons) |
| **Classical ML** | Tabular model with time-aware CV and calibrated probabilities (game outcomes) |
| **Unsupervised ML** | Representation learning and clustering of player styles across eras |
| **Deep learning** | Sequence model (LSTM) over player careers |
| **Product** | An interactive dashboard that makes all of the above explorable |

Guiding ideas found in the code:
- **Cross-era comparability:** z-score stats *within each season* so a 1985 player and a 2025 player
  can be compared fairly (the pace and 3-point eras differ hugely).
- **One shared feature module** (`ml/features.py`). All models import it so the features stay
  consistent between training and inference.
- **Every model is a CLI-trainable module** (`python -m ml.<model> train`). It saves a single bundle
  to `models/` and exposes plain Python functions (`predict_winner`, `find_similar`,
  `predict_next_season`) that the app calls.
- **Polite, resumable scraping:** sleep between requests, checkpoint per season, and
  `ON CONFLICT DO NOTHING` so any season can be re-run safely.

### 1.1 Original roadmap (reconstructed)

Only Phase 1 was ever written down (`PHASE1_ETL_GUIDE.md`, probably produced in a separate chat).
There was no document for later phases, so the plan below comes from three sources: the hand-off
notes in that guide, the scaffolding created on day one, and file timestamps. Each row says how
solid its evidence is.

| Phase | Scope | Evidence | Status |
|---|---|---|---|
| **0: Prototype** | `scrapping.py`: generic Basketball-Reference table fetcher, player totals and per-game, conference standings | File dated 2026-01-12, four months before the rest | ✅ done |
| **1: ETL & database** | Postgres setup, `schema.sql`, ORM models, `etl/scraper.py` (advanced stats, both-conference standings, monthly schedule), idempotent `loader.py`, resumable `pipeline.py` for 1980–2025, verification SQL | **Documented** in `PHASE1_ETL_GUIDE.md` | ✅ built 2026-05-15/16, but the data has the bugs in §7 |
| **2: Game predictor** | Win-probability model on `games` + `team_season_stats` | **Documented in the guide's hand-off:** *"Once Phase 1 is done and verified … we'll build Phase 2 (game predictor model)"* | ✅ trained 2026-05-19 12:50, invalid because of §7.1–7.3 |
| **3 (inferred): Player similarity & archetypes** | PCA → UMAP → KMeans, "who plays like X?" | `umap-learn` was in `requirements.txt` from day one; module trained 2026-05-19 12:25–12:34, *before* the game predictor | ✅ trained |
| **4 (inferred): Career-arc LSTM** | Next-season stat projection | `torch` was in `requirements.txt` from day one; trained 2026-05-19 14:28–14:52. `features.py` was last touched at 14:51, so the shared feature module was being adjusted for it | ✅ trained |
| **5 (inferred): Streamlit dashboard** | Multi-page app consuming the DB and all three models | The guide says *"When the ML models and Streamlit app are built, they'll connect to this database"*. `app/pages/` was created on day one (2026-05-15 14:35) and `run.sh` points to `app/Home.py`. `streamlit` and `plotly` were in `requirements.txt` from the start | ⏳ only `app/utils.py` was built (2026-05-19 17:44); no pages |
| **Planned but unused** | `notebooks/` for EDA (created empty on day one); `pyyaml` in the guide's install line (a config file was probably planned and never used); `bref_id`, `position` and `birth_year` columns in `players` (designed in, never populated) | Scaffolding and schema | ❌ not started |

Build timeline: prototype 2026-01-12 → scaffolding and guide 2026-05-15 → ETL run 2026-05-16 →
all three models and app helpers on 2026-05-19, in the order similarity → game predictor → LSTM →
app utils. After that the source `.py` files were deleted; folder modification dates point to
mid-July and mid-August 2026.

How the phases were organised: each phase was meant to be fully verified before the next one started.
Phase 1 ended on verification SQL and *"Once Phase 1 is done and verified, shout"*. That check was
skipped or too shallow, which is how the inverted scores and missing standings reached the models.
The validation script in §8 Phase 1 step 5 is meant to bring that gate back.

---

## 2. Architecture

```
┌────────────────────────┐   HTTP (4–5 s sleep/request)
│ basketball-reference   │◄──────────────────────────────┐
└────────────────────────┘                               │
                                                         │
┌──────────────── etl/ ──────────────────────────────────┴─────┐
│ scraper.py   fetch_league_table(url, table_id)  (+ HTML-comment │
│              fallback), per-table wrappers                      │
│ loader.py    idempotent INSERT … ON CONFLICT DO NOTHING         │
│ pipeline.py  season loop 1980–2025, checkpoint file             │
└───────────────────────────────┬────────────────────────────────┘
                                ▼
┌──────────────── PostgreSQL 17 (nba_platform) ─────────────────┐
│ players · player_season_stats · team_season_stats · games     │
└───────────────────────────────┬───────────────────────────────┘
                 db/connection.py: get_engine() (lru_cached), query_df(sql, params)
                                ▼
┌──────────────── ml/ ──────────────────────────────────────────┐
│ features.py        loaders + feature builders (shared)          │
│ game_predictor.py  XGBoost + isotonic calibration → .pkl        │
│ similarity.py      z-score → PCA(10) → UMAP(2) → KMeans(9)      │
│ player_lstm.py     2-layer LSTM, seq of 4 seasons → next season │
└───────────────────────────────┬───────────────────────────────┘
                                ▼  models/*.pkl, *.pt
┌──────────────── app/ (Streamlit) ─────────────────────────────┐
│ Home.py + pages/*  (NOT BUILT)                                  │
│ utils.py  cached queries, Plotly chart helpers, dark theme      │
└───────────────────────────────────────────────────────────────┘
run.sh:  PYTHONPATH=. streamlit run app/Home.py
```

### Tech stack
- Python 3.11 (conda env `nba`)
- Scraping: `requests`, `beautifulsoup4`, `pandas.read_html`
- DB: PostgreSQL 17 (Homebrew), `sqlalchemy>=2`, `psycopg2-binary`, `python-dotenv`
- ML: `scikit-learn`, `xgboost`, `umap-learn`, `torch` (MPS on Apple Silicon → CUDA → CPU), `joblib`
- App: `streamlit`, `plotly`
- Utility: `tqdm`

### Folder layout (target)
```
BasketBall/
├── .env / .env.example      DATABASE_URL=postgresql://nba_user:<pw>@localhost:5432/nba_platform
├── requirements.txt
├── run.sh
├── db/   schema.sql, models.py (ORM mirror), connection.py
├── etl/  scraper.py, loader.py, pipeline.py
├── ml/   features.py, game_predictor.py, similarity.py, player_lstm.py
├── app/  Home.py, utils.py, pages/
├── models/      (gitignored artifacts)
└── notebooks/   (EDA, empty so far)
```

---

## 3. Data Layer

### 3.1 Source: Basketball-Reference
`season` convention: **the ending year**, so `2024` means the 2023-24 season. Range: `1980..2025`.

| Data | URL | table id |
|---|---|---|
| Per-game | `/leagues/NBA_{y}_per_game.html` | `per_game_stats` |
| Totals | `/leagues/NBA_{y}_totals.html` | `totals_stats` |
| Advanced | `/leagues/NBA_{y}_advanced.html` | `advanced_stats` |
| Standings | `/leagues/NBA_{y}_standings.html` | `confs_standings_E` / `confs_standings_W` (see bug §7.3) |
| Schedule | `/leagues/NBA_{y}_games-{month}.html` | `schedule` (one page per month, Oct→Jun; no October before 2002) |

Core scraper. Keep this, because it works:

```python
def fetch_league_table(url, table_id, sleep_s=4.0, clean_data=True) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; personal-research)",
               "Accept-Language": "en-US,en;q=0.9"}
    r = requests.get(url, headers=headers, timeout=30); r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"id": table_id})
    if table is None:                      # BRef hides many tables inside HTML comments
        for c in soup.find_all(string=lambda t: t and "<table" in t):
            if f'id="{table_id}"' in c:
                table = BeautifulSoup(c, "html.parser").find("table", {"id": table_id}); break
    if table is None:
        raise ValueError(f"Table id='{table_id}' not found")
    df = pd.read_html(str(table))[0]
    if clean_data:
        # flatten MultiIndex headers; drop repeated header rows (Rk == 'Rk'); reset index;
        # coerce numeric columns except Player/Pos/Team
        ...
    time.sleep(sleep_s)
    return df
```

Rate limits: Basketball-Reference allows about 20 requests/min and bans offenders for hours.
Keep the sleep at ≥ 3–4 s (5 s for schedule pages). A full run is about 46 seasons × ~13
requests, which takes roughly 3–5 hours.

### 3.2 Schema (`db/schema.sql`)

```sql
CREATE TABLE IF NOT EXISTS players (
    id SERIAL PRIMARY KEY, name TEXT NOT NULL,
    bref_id TEXT UNIQUE,             -- intended as the real identity, NEVER populated (bug §7.4)
    position TEXT, birth_year INT
);
CREATE TABLE IF NOT EXISTS player_season_stats (
    id SERIAL PRIMARY KEY,
    player_id INT REFERENCES players(id),
    team_abr  TEXT NOT NULL,
    season    INT  NOT NULL,
    stat_type TEXT NOT NULL,         -- 'per_game' | 'totals' | 'advanced'  (long format)
    g FLOAT, gs FLOAT, mp FLOAT, pts FLOAT, reb FLOAT, ast FLOAT, stl FLOAT, blk FLOAT,
    tov FLOAT, pf FLOAT, fg FLOAT, fg_pct FLOAT, fg3 FLOAT, fg3_pct FLOAT, ft FLOAT, ft_pct FLOAT,
    per FLOAT, ts_pct FLOAT, usg_pct FLOAT, ws FLOAT, bpm FLOAT, vorp FLOAT,
    UNIQUE (player_id, season, team_abr, stat_type)
);
CREATE TABLE IF NOT EXISTS team_season_stats (
    id SERIAL PRIMARY KEY, team_abr TEXT NOT NULL, season INT NOT NULL, conference TEXT,
    w INT, l INT, w_pct FLOAT, gb FLOAT, ps_g FLOAT, pa_g FLOAT,
    UNIQUE (team_abr, season)
);
CREATE TABLE IF NOT EXISTS games (
    id SERIAL PRIMARY KEY, season INT NOT NULL, game_date DATE NOT NULL,
    home_team TEXT NOT NULL, away_team TEXT NOT NULL,
    home_score INT, away_score INT, home_win BOOLEAN,
    UNIQUE (season, game_date, home_team, away_team)
);
CREATE INDEX IF NOT EXISTS idx_pss_player ON player_season_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_pss_season ON player_season_stats(season);
CREATE INDEX IF NOT EXISTS idx_pss_type   ON player_season_stats(stat_type);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);
CREATE INDEX IF NOT EXISTS idx_games_date   ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_tss_season   ON team_season_stats(season);
```

Design decisions:
- **Long format for player stats** (one row per `stat_type`) instead of one wide row. It is simpler
  to load from three separate pages, but every consumer has to pivot or merge (see
  `build_similarity_features`). *Consider a single wide `player_seasons` table in the rebuild.*
- `db/models.py` is a SQLAlchemy ORM mirror of the schema, but **nothing uses it**. All code uses
  raw SQL via `query_df`. Either drop it or use it.
- `db/connection.py`: `get_engine()` is `lru_cache`'d with `pool_pre_ping=True`.
  `query_df(sql, params)` returns a DataFrame through `pd.read_sql_query(text(sql), …)`.

### 3.3 Loader (`etl/loader.py`)
- Column mapping BRef → DB: `Player→name, Pos→position, Tm→team_abr, G, GS, MP, PTS, TRB→reb, AST,
  STL, BLK, TOV, PF, FG, FG%→fg_pct, 3P→fg3, 3P%→fg3_pct, FT, FT%→ft_pct, PER, TS%→ts_pct,
  USG%→usg_pct, WS, BPM, VORP`.
- **Traded players:** if a player has a `TOT` row, keep only the `TOT` row and drop the per-team rows,
  giving one row per player-season.
- `_get_or_create_player(conn, name)`: SELECT by **name**, otherwise INSERT … RETURNING id.
- `_flt(row, col)`: safe float conversion returning None on NaN or bad values.
- Missing team → `'UNK'`.
- Standings: the team column is detected dynamically (the first column whose name contains
  "conference" or equals "Team"), `*` is stripped from team names (it marks a playoff team), and
  rows with no team are dropped. `gb` is never loaded.
- All functions return `rowcount` so the pipeline can print "N rows inserted".
- Uses `engine.begin()` (one transaction per DataFrame) and row-by-row `execute`. This is fine at
  this scale (about 20k rows per stat type).

### 3.4 Pipeline (`etl/pipeline.py`, run as `python -m etl.pipeline [--season Y | --start Y]`)
- Steps per season: per_game → totals → advanced → standings → schedule. Each step is wrapped in
  try/except so one failing step does not kill the season.
- Checkpoint: `.etl_checkpoint.json` holds the list of completed seasons, and those seasons are
  skipped on re-run.
- ⚠️ A season is marked "done" **even if steps failed**, which is how the broken standings went
  unnoticed. In the rebuild, checkpoint per *(season, step)* and only on success.

### 3.5 Current DB contents (verified 2026-09-30)
| Table | Rows | Notes |
|---|---|---|
| players | 3,690 | `bref_id` NULL for all |
| player_season_stats | 19,990 × 3 stat types | seasons 1980–2025; **team_abr = 'UNK' for all 59,970 rows** |
| team_season_stats | **163** (should be about 1,300) | only 1980 and 2016–2025, **East only** |
| games | 54,555 | 1980–2025; **home/away scores swapped** |

---

## 4. ML Layer (recovered from bytecode)

### 4.1 `ml/features.py`: shared feature engineering
```python
load_player_stats(stat_type="per_game", min_games=20)  # joins players; filters g >= min_games
load_team_stats()   # team_abr, season, conference, w, l, w_pct, ps_g, pa_g
load_games()        # ordered by game_date

SIMILARITY_COLS = ['pts','reb','ast','stl','blk','tov','fg_pct','fg3_pct','ft_pct',
                   'ts_pct','usg_pct','per','bpm']
LSTM_STAT_COLS  = ['pts','reb','ast','stl','blk','fg_pct']
```

**`build_game_features(games, teams)`**
- `teams.pts_diff = ps_g - pa_g`, prefixed `home_*` and `away_*`, then merged onto games by
  `(season, team name)`.
- It also computes a 10-game rolling win% per team (`rolling(10, min_periods=3).mean().shift(1)`),
  **but the result is appended to a list that is never merged in. This is dead code** (the right
  idea, never wired up).
- Features: `home_w_pct, away_w_pct, home_pts_diff, away_pts_diff, home_ps_g, home_pa_g, away_ps_g, away_pa_g`
- `target = home_win.astype(int)`. Rows with any NaN feature are dropped.

**`build_similarity_features(min_games=41)`**
- per_game LEFT JOIN advanced on `(name, season, team_abr)`; `per, ts_pct, usg_pct, bpm` come from advanced.
- **Per-season z-score**: `(x - mean_season) / (std_season + 1e-8)` for each SIMILARITY_COL.
- Returns `name, season, team_abr, position` + the 13 z-scored cols.

**`build_career_sequences(min_seasons=SEQ_LEN)`**
- per_game with `min_games=20`; **global** z-score of LSTM_STAT_COLS (not per season).
- For each player sorted by season: sliding windows of length `SEQ_LEN=5`. The first 4 seasons are X
  and the 5th is y.
- Returns `X (N, 4, 6)`, `y (N, 6)` as float32, plus `names`.

### 4.2 `ml/game_predictor.py`: XGBoost win probability
- CV: `TimeSeriesSplit(n_splits=5)` over games sorted by date; reports accuracy and ROC-AUC per fold.
- Model: `XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
  colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss", random_state=42)`
- Final model: `StandardScaler` → `CalibratedClassifierCV(xgb, cv=5, method="isotonic")` on the full data.
- Bundle `models/game_predictor.pkl`: `{model, scaler, feature_cols, train_accuracy, train_auc}`.
  "train_*" actually holds the **mean CV** metrics.
- `predict_winner(home_team, away_team, season)`: uses the team's row for that season, or else its
  latest season. Missing values default to `w_pct=0.5` and `ps_g/pa_g=110`. Returns
  `{home_team, away_team, home_win_prob, away_win_prob, predicted_winner, confidence, model_accuracy}`.
  Confidence is `high` if |p-0.5|>0.15, `moderate` if >0.07, otherwise `low`.
- `get_model_stats()` → `{accuracy, auc, features}` for the dashboard.

### 4.3 `ml/similarity.py`: player similarity & archetypes
- Pipeline: `build_similarity_features(min_games=41)` → `PCA(n_components=10, random_state=42)` →
  `umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric="cosine", random_state=42)` on the
  PCA output → `KMeans(n_clusters=9, n_init=20, random_state=42)` on the PCA output.
- Archetype labels, hard-coded **by cluster index**: `0 '3-and-D Wing', 1 'Pass-First Guard',
  2 'Scoring Guard', 3 'Perimeter Scorer', 4 'Athletic Slasher', 5 'Floor General',
  6 'Traditional Big', 7 'Modern Stretch Big', 8 'Two-Way Anchor'`.
- Bundle `models/similarity.pkl` (~15 MB): `{df (with umap_x, umap_y, cluster, archetype), X_pca, pca,
  reducer, kmeans, feature_cols, archetype_names}`.
- API:
  - `find_similar(player_name, season=None, top_k=10, cross_era=True)`: cosine similarity in PCA
    space, excludes the player themself, and with `cross_era=False` restricts to ±5 seasons. Returns
    `name, season, team_abr, archetype, similarity`.
  - `get_archetypes()`, `get_player_archetype(name, season=None)` (latest season if None),
    `get_umap_plot_data()` → `name, season, team_abr, archetype, umap_x, umap_y, pts, reb, ast`.

### 4.4 `ml/player_lstm.py`: career-arc LSTM
- Hyperparameters: `SEQ_LEN=5` (4 in → 1 out), `HIDDEN_DIM=128`, `N_LAYERS=2`, `DROPOUT=0.3`,
  `LR=1e-3`, `weight_decay=1e-5`, `EPOCHS=80`, `BATCH_SIZE=64`.
- Model:
  ```python
  class CareerLSTM(nn.Module):
      def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout):
          self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True,
                              dropout=dropout if n_layers > 1 else 0.0)
          self.dropout = nn.Dropout(dropout)
          self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                                    nn.Linear(hidden_dim // 2, output_dim))
      def forward(self, x):
          out, _ = self.lstm(x)
          return self.head(self.dropout(out[:, -1, :]))
  ```
- Training: random 85/15 split (seed 42), Adam, `ReduceLROnPlateau(patience=8, factor=0.5)`, MSE,
  grad-clip 1.0, keeps the best-val state dict, and raises if every loss is NaN. Device order:
  mps → cuda → cpu.
- Artifacts: `models/player_lstm.pt` (state dict) and `models/player_lstm_meta.pkl` =
  `{n_features: 6, stat_cols, means, stds, best_val_loss: 0.256}` (normalised MSE).
- `predict_next_season(name)` → `{player, based_on_seasons, predicted_next_season: {stat: value}}`.
  Returns None if the player has fewer than 4 seasons.

---

## 5. App Layer (Streamlit)

Only `app/utils.py` was ever written (recovered from bytecode). `app/Home.py` and `app/pages/` do
not exist.

`app/utils.py` contents:
- `@st.cache_data` queries: `cached_player_list()`, `cached_team_list()`, `cached_season_range()`,
  `cached_player_career(name)` (per_game + advanced cols by season),
  `cached_team_history(team)`, `cached_leaders(season, stat="pts", n=…)` (g ≥ 41),
  `cached_league_averages()` (league avg pts/reb/ast/fg3_pct/ts_pct/bpm per season, g ≥ 41).
- Plotly helpers: `career_line(df, y_col, title, y_label)`, `stat_bar(df, x_col, y_col, title)`.
  Accent `#2563eb`, plot bg `#0f0f0f`, text `#e5e7eb`, grid `#1f2937`.
- `apply_dark_theme()` injects CSS (app bg `#0a0a0a`, sidebar `#111111`, `.metric-card` styles), and
  `metric_card(label, value)` renders an HTML stat tile.
- ⚠️ `cached_leaders` builds SQL by f-string with `stat` as a column name. Whitelist `stat`.

---

## 6. Key Design Decisions (and whether to keep them)

| Decision | Rationale | Keep? |
|---|---|---|
| Basketball-Reference HTML scraping | Free, deep history (1980+), consistent tables | ✅ but use player `bref_id`s from `data-append-csv` on the player cell |
| PostgreSQL rather than SQLite or files | "Real" data-engineering stack, SQL queries from the app | ✅ (SQLite or DuckDB would also be fine) |
| Idempotent upserts + checkpoint | Long scrape and polite rate limits mean it must be resumable | ✅ improve the checkpoint granularity |
| Keep only the `TOT` row for traded players | One row per player-season simplifies the ML | ✅ but also store per-team rows for team views |
| Per-season z-score for similarity | Era adjustment (pace, 3PT revolution) | ✅ core idea |
| PCA → KMeans; UMAP only for 2-D viz | Denoise, then cluster; UMAP for the scatter | ✅ |
| `TimeSeriesSplit` + isotonic calibration | No future leakage in CV; meaningful probabilities | ✅ but the features themselves leak (§7.2) |
| LSTM on 4-season windows | Showcases deep learning on sequences | ✅ but compare it against simple baselines |
| Separate train CLI per model + joblib bundle | Simple, reproducible, app just loads | ✅ |
| Streamlit + Plotly dark theme | Fast to build a multi-page dashboard | ✅ |

---

## 7. Verified Bugs & Pitfalls (fix these in the rebuild)

### 7.1 🔴 Game scores and home/away results are inverted
BRef schedule columns are `Date, Start, Visitor/Neutral, PTS, Home/Neutral, PTS.1, …`. So the first
`PTS` belongs to the **visitor**. The code mapped `PTS→home_score` and `PTS.1→away_score`.
Evidence: the stored home-win rate is **39.4%** (the NBA's real figure is about 58–60%). The
2023-10-24 LAL @ DEN game is stored as a Denver 107–119 loss when Denver won 119–107.
**Fix:** `PTS→away_score`, `PTS.1→home_score`. The trained game model learned flipped labels and is
unusable.

### 7.2 🔴 Game predictor target leakage
Features are the **same season's final** standings (`w_pct`, `ps_g`, `pa_g`). Those include the game
being predicted and every later game, so CV accuracy is inflated.
**Fix:** use only information available *before* tip-off: rolling or expanding win% and point
differential from the games table (shifted by one game), previous-season stats for early-season
games, rest days or back-to-backs, and Elo ratings. The rolling-win% code already exists but was
never merged (§4.1).

### 7.3 🔴 Standings are mostly missing
Only East rows were loaded, and only for 1980 and 2016–2025. Two causes:
- The East and West tables have different first-column headers ("Eastern Conference" and "Western
  Conference"). After `concat`, the loader renames the *first* matching column to `team_abr` and
  drops NaN, which wipes out every West row.
- Before about 2016 the standings page shows **division** tables (`divs_standings_E/W`), so
  `confs_standings_*` is not found.

Knock-on effect: the game model trained only on games where both teams had standings rows.
**Fix:** normalise each table's team column *before* concatenating, and fall back to
`divs_standings_*` (dropping division header rows). Better still, derive W/L, PS/G and PA/G directly
from the `games` table, which is complete.

### 7.4 🔴 Player team is `UNK` everywhere, and identity is by name
- Every `player_season_stats.team_abr` is `'UNK'`: the scraped table no longer has a `Tm` column.
  Basketball-Reference renamed it (likely `Team`), and multi-team rows are now likely `2TM`/`3TM`
  rather than `TOT`. Verify against the live HTML. As a result the TOT de-duplication never ran.
  Only the first row per player-season survived, because of the
  `UNIQUE(player_id, season, team_abr, stat_type)` + `ON CONFLICT DO NOTHING` combination.
- Players are keyed by **name only**, and `bref_id` is never filled. Different players with the same
  name (e.g. two "Tony Mitchell"s) are merged into one career, which corrupts the LSTM and
  similarity data.
- Name formatting: strip the trailing `*` (Hall of Fame marker) if present, and normalise accents.
**Fix:** parse `data-append-csv` (the BRef player id) from the `<td data-stat="name_display|player">`
cell, and make `bref_id` the identity. Map `Team`/`Tm`, and treat `TOT|2TM|3TM|4TM` as the combined row.

### 7.5 🟠 Team naming inconsistency
`games` and standings store **full names** ("Boston Celtics"), but the docs and API examples use
abbreviations (`predict_winner("LAL", "BOS")`). Franchises also relocate or rename (Seattle → OKC,
New Jersey → Brooklyn, Charlotte Bobcats/Hornets…).
**Fix:** add a `teams` table with `(franchise_id, abbreviation, full_name, first_season, last_season)`
and a name→abbreviation mapping.

### 7.6 🟠 Similarity issues
- Archetype names are attached to **KMeans cluster IDs arbitrarily**. Cluster 0 is not guaranteed to
  be "3-and-D Wing". Name clusters after inspecting the centroids (top z-scores per cluster), or
  assign names with a rule-based mapping from the centroid profile.
- `find_similar` docstring says `season=None` means the career average, but the code uses the
  **latest** season.
- The `pts/reb/ast` returned by `get_umap_plot_data` are **z-scores**, not raw values. Keep the raw
  columns for tooltips.
- `StandardScaler` is imported but never used (fine, since the per-season z-score already happens).

### 7.7 🟠 LSTM methodology
- The random train/val split is over *windows*, so windows from the same player end up in both sets
  (leakage). Split by player instead, or by time (train on targets ≤ 2018, validate on later ones).
- Consecutive rows are assumed to be consecutive seasons. Gaps (injury or overseas seasons, or
  seasons filtered by `g ≥ 20`) are ignored. Add an age or season-gap feature, or skip windows with gaps.
- Only 6 targets and no age input, though age is the strongest aging-curve signal.
  Add `birth_year` → age.
- There is no baseline. Compare against "same as last season" and a weighted-average or Marcel-style
  projection. `best_val_loss = 0.256` (normalised MSE) means nothing without one.

### 7.8 🟡 Engineering hygiene
- Not a git repo; no tests; `notebooks/` empty; `etl/scraper.py` source missing (it exists only in
  `PHASE1_ETL_GUIDE.md`); ML source survives only as `.pyc`.
- `clean_basketball_reference_data` uses `pd.to_numeric(errors="ignore")`, which is deprecated. Use
  an explicit coerce over known numeric columns instead.
- `requirements.txt` has no pinned versions. The env that worked was: Python 3.11, sklearn 1.8,
  xgboost 3.2, torch 2.12, umap-learn 0.5.12, streamlit 1.57, plotly 6.7.
- XGBoost ≥ 2 dropped `use_label_encoder`, so remove it.

---

## 8. Rebuild Plan (recommended order)

**Phase 0: Setup**
1. `git init`, conda env (py3.11), Postgres 17 via Homebrew, `.env` with `DATABASE_URL`.
2. Pin requirements. Add `pytest` and `ruff`.

**Phase 1: ETL (fix §7.1, §7.3, §7.4, §7.5)**
1. `etl/scraper.py`: `fetch_league_table` + wrappers. Extract `bref_id` from player cells. Handle the
   `Team`/`Tm` and `TOT`/`2TM` variants. Fix the schedule PTS mapping. Add division-standings fallback.
2. Optional: cache raw HTML to `data/raw/{season}/{page}.html` so parsing can be re-run without
   re-scraping. This is strongly recommended given the rate limits.
3. `etl/loader.py`: upsert by `bref_id`. Add a franchise and team mapping table.
4. `etl/pipeline.py`: checkpoint per (season, step) only on success; add a `--refresh` flag.
5. **Validation script** (`etl/validate.py`) asserting:
   - home-win rate is between 0.55 and 0.65
   - each season has 23–30 teams in standings, both conferences
   - `team_abr != 'UNK'`
   - about 400–600 player rows per season
   - the 1999 and 2012 lockout seasons have fewer games; 2020 has the bubble
   - spot-check known games and box totals

**Phase 2: Features & Models**
1. `ml/features.py`: leak-free game features (pre-game rolling/expanding win% and net rating,
   previous-season fallback, rest days, home/away splits, Elo).
2. Game predictor: same XGBoost + TimeSeriesSplit + isotonic. Report log-loss and Brier as well, and
   compare against baselines (always-home ≈ 60%, Elo-only).
3. Similarity: same pipeline. Name clusters from centroids, and add a career-average mode.
4. LSTM: player-level or temporal split, age feature, baselines, and per-stat MAE in real units.

**Phase 3: Streamlit app** (§9)

---

## 9. Next Steps / Not Done Yet

### Never built
- [ ] **`app/Home.py` and every page in `app/pages/`.** The dashboard is the product and doesn't exist yet. Intended pages, inferred from `utils.py` and the model APIs:
  - **Home:** overview metric cards (seasons, players, games, model accuracy) and league trend lines (`cached_league_averages`: the rise of the 3-pointer, TS%).
  - **Player Profile:** career lines (`career_line`), archetype badge (`get_player_archetype`), similar players (`find_similar`, with the cross-era toggle), next-season projection (`predict_next_season`).
  - **Team History:** W/L and net rating over time (`cached_team_history`).
  - **Leaders:** per-season leaderboards for any stat (`cached_leaders` + `stat_bar`).
  - **Game Predictor:** pick home and away team + season → win probability gauge and confidence.
  - **Archetype Explorer:** UMAP scatter coloured by archetype, filterable by season and era, with hover tooltips.
- [ ] `notebooks/`: EDA (era trends, data-quality checks) and model evaluation notebooks.
- [ ] Tests (scraper parsing on saved HTML fixtures, loader idempotency, feature leakage checks).
- [ ] README with screenshots.

### Must fix (data and model validity)
- [ ] Swap home and away scores (§7.1), then re-scrape the schedule or run a one-off SQL fix
      (`UPDATE games SET home_score=away_score, away_score=home_score, home_win=NOT home_win`).
- [ ] Re-scrape standings for both conferences and all seasons, or derive them from `games` (§7.3).
- [ ] Re-scrape player stats with `bref_id` and real teams (§7.4).
- [ ] Rebuild the game features without leakage and retrain (§7.2).
- [ ] Name the archetype clusters from their centroids (§7.6).
- [ ] Fix the LSTM split, add age, add baselines (§7.7).

### Nice-to-have extensions
- [ ] Playoff games (separate pages) and a playoff-series simulator built on the game model.
- [ ] Player game logs → in-season form features; shot-location data (more recent seasons).
- [ ] Incremental daily update job for the current season (only scrape new months).
- [ ] SHAP explanations for the game predictor in the UI.
- [ ] Dockerise (Postgres + app) and deploy (Streamlit Community Cloud plus a hosted Postgres such as Neon or Supabase).
- [ ] MLflow or simple JSON metrics logging for model versions.

---

## Appendix A: Useful verification SQL
```sql
SELECT stat_type, COUNT(*), MIN(season), MAX(season) FROM player_season_stats GROUP BY 1;
SELECT AVG(home_win::int) FROM games;                       -- expect ~0.58–0.62
SELECT season, COUNT(*), string_agg(DISTINCT conference, ',') FROM team_season_stats GROUP BY 1 ORDER BY 1;
SELECT team_abr, COUNT(*) FROM player_season_stats GROUP BY 1 ORDER BY 2 DESC LIMIT 5;  -- no 'UNK'
SELECT p.name, s.pts FROM player_season_stats s JOIN players p ON p.id=s.player_id
WHERE s.season=2024 AND s.stat_type='per_game' ORDER BY s.pts DESC LIMIT 10;
```

## Appendix B: Run commands (original conventions)
```bash
psql $DATABASE_URL -f db/schema.sql
python -m etl.pipeline --season 2024     # smoke test one season
python -m etl.pipeline                   # full history (resumable)
python -m ml.game_predictor train
python -m ml.similarity train            # UMAP ≈ 60 s
python -m ml.player_lstm train
bash run.sh                              # PYTHONPATH=. streamlit run app/Home.py
```
