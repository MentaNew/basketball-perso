# Phase 5 — Streamlit Dashboard

Make the data and the three models explorable in one dark-themed, multi-page app. v1 never
got past `app/utils.py`. This phase is the product.

Files: `app/Home.py`, `app/utils.py`, `app/pages/*.py`, `run.sh`.

---

## Step 0 — Architecture

```
                     ┌──────────── app/Home.py + app/pages/*.py ────────────┐
                     │  layout, widgets, charts; no SQL, no model internals  │
                     └───────────────┬───────────────────────┬──────────────┘
                                     │                       │
              app/utils.py ◄─────────┘                       └────────► ml/*.py public APIs
   cached SQL queries (st.cache_data)                         predict_winner, find_similar,
   Plotly helpers, theme, metric cards                        predict_next_season, …
                │                                                      │
                ▼                                                      ▼
        PostgreSQL (query_df)                                  models/*.pkl, *.pt
```

Two rules keep the pages thin:
1. **Pages never write SQL.** Every query lives in `app/utils.py` as a `cached_*` function.
2. **Pages never touch model internals.** They call the public functions from Phases 2–4.
   Each model module loads its bundle lazily, once per process (`_load()`).

Launch:

```bash
./run.sh            # = PYTHONPATH=. streamlit run app/Home.py
```

`PYTHONPATH=.` is what makes `from app.utils import …` and `from ml.… import …` work.
Streamlit runs each page file as a script, not as a package module.

---

## Step 1 — `app/utils.py`

### 1.1 Cached queries

All are decorated with `@st.cache_data(ttl=3600)`, so each one hits Postgres once per hour
per argument set.

| Function | Returns | Used by |
|---|---|---|
| `cached_counts()` | players, games, seasons, franchises | Home |
| `cached_league_averages()` | per-season 3PA rate, 3P%, TS% (minutes-weighted, g ≥ 41) | Home |
| `cached_league_pace()` | per-season combined points and home-win rate | Home |
| `cached_player_list()` | bref_id, name, career span, and a disambiguating `label` | Player Profile |
| `cached_player_career(bref_id)` | one primary row per season | Player Profile |
| `cached_team_list(season=None)` | franchises (latest name), or that season's teams | Team History, Game Predictor |
| `cached_team_history(franchise_id)` | all seasons of a franchise + `net_rating` | Team History |
| `cached_leaders(season, stat, n)` | top-N for one stat | Leaders |
| `cached_season_range()` | (min, max) season | Leaders |

**Security note (§5 in the blueprint):** `cached_leaders` puts a *column name* into SQL, and
column names can't be bind parameters. v1 f-stringed user input straight in. Now `stat` must
be a key of the `LEADER_STATS` whitelist, or the function raises `ValueError`. Percentage
stats also need minimum attempts (`PCT_MIN_ATTEMPTS`), so a 2-for-2 shooter can't lead 3P%.

Player selection uses `bref_id` internally. The dropdown shows a unique `label`: name and
career span, plus teams when namesakes share a span. Both Tony Mitchells played only in 2013-14,
so they appear as "Tony Mitchell (2013–2014, DET)" and "Tony Mitchell (2013–2014, MIL)".

### 1.2 Chart helpers

| Helper | Use |
|---|---|
| `style_fig(fig, height)` | Applies the dark theme to any Plotly figure. Call it on everything |
| `career_line(df, y_col, title)` | Season-by-season line with team/age/games on hover |
| `stat_bar(df, x, y, title)` | Horizontal leaderboard bars, height scaled to row count |
| `win_gauge(prob, label)` | Gauge for win probability |

Theme tokens: accent `#2563eb`, plot background `#0f0f0f`, text `#e5e7eb`, grid `#1f2937`.
`ARCHETYPE_COLORS` is a fixed qualitative palette, so an archetype keeps its colour across
charts.

### 1.3 Layout helpers

- `apply_dark_theme()`: injects CSS (app background `#0a0a0a`, sidebar `#111111`,
  `.metric-card`, `.badge`). Call it at the top of every page.
- `metric_card(label, value)`: a stat tile.
- `model_available("x.pkl")`: lets a page show "train the model first" instead of crashing
  when an artifact is missing.
- `season_label(2024)` returns `"2023-24"`. Seasons are stored as ending years and shown
  the NBA way.

---

## Step 2 — Pages

Every page starts with the same three lines:

```python
st.set_page_config(page_title="…", page_icon="🏀", layout="wide")
apply_dark_theme()
st.title("…")
```

### Home (`app/Home.py`)
- Metric cards: seasons, players, games, and held-out game-model accuracy (if trained).
- Four league-trend charts: the three-point revolution (3PA rate), true shooting %, combined
  points per game, and home-court advantage (note the 2020 bubble dip to ~50%).

### 1 · Player Profile
- Career metric cards (games-weighted career PPG/RPG/APG, total win shares) and the latest
  **archetype badge** (`get_player_archetype`).
- Tabs:
  - **Career arc:** six `career_line` charts (PTS, REB, AST, TS%, PER, BPM).
  - **Similar players:** `find_similar` with a one-season / whole-career switch, a cross-era
    toggle, a result count, and a season slider limited to 41+-game seasons.
  - **Next-season projection:** `predict_next_season`, last season vs. projected with deltas.
    Explains when there isn't enough history.
  - **Season table:** the raw career rows.

### 2 · Team History
- Franchise picker (relocations merged via `franchise_id`).
- Cards: seasons, all-time W–L, playoff appearances, best season.
- Wins per season (bars, with ★ for playoff teams), and point differential vs. SRS.

### 3 · Leaders
- Season slider, a stat chosen from the whitelist, and Top N, rendered as a horizontal bar
  chart. Traded players appear on their combined line, labelled with all their teams.

### 4 · Game Predictor
- "Team strength as of the end of" season slider, home/away pickers limited to that season's
  teams, and **Playoff game** / **Neutral site** toggles. With a neutral site, the gauges drop
  the "(home)"/"(away)" labels and Elo drops home court.
- Two win-probability gauges, then predicted winner, confidence, and the Elo-only probability
  for comparison.
- "How good is the model?" expander: the held-out comparison table vs. baselines, plus
  feature importance.

### 5 · Archetype Explorer
- UMAP scatter coloured by archetype, with a season-range filter and raw-stat hover tooltips.
- "Highlight player" search overlays a player's seasons, labelled by year.
- Archetype mix by season (stacked area). "3-and-D Wing" grows steadily from ~9% of
  qualifying players in 1985 to ~20% in 2025.
- Centroid table, so users can see *why* a cluster has its name.

---

## Step 3 — Run & verify

```bash
./run.sh                                    # opens http://localhost:8501
```

Headless smoke test that renders every page and exercises the main widgets:

```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app/pages/1_Player_Profile.py", default_timeout=180).run()
assert not at.exception
at.radio[0].set_value("Whole career").run()
assert not at.exception
```

Checklist:
- [ ] Every page loads with no models trained (it shows an info box, not a traceback).
- [ ] Every page loads with all models trained.
- [ ] Player Profile → Similar players → Whole career works with cross-era off.
- [ ] Game Predictor: Elo-only and model probabilities are in the same ballpark, and the
      Neutral-site toggle lowers the home team's Elo-only probability.

---

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: app` / `ml` | Launched without `PYTHONPATH=.` | Use `./run.sh` |
| Page hangs on the LSTM projection after using the predictor | OpenMP clash between xgboost and torch | Keep `torch.set_num_threads(1)` in `player_lstm._load()` |
| New data or model doesn't show up | `st.cache_data` (1 h) and module-level bundle caches | Restart the server, or use "Clear cache" in the ⋮ menu |
| First load of Explorer/Profile is slow | Unpickling the similarity bundle | Only the first request in each process pays this |

---

## What Comes Next

See `NEXT_STEPS.md`: deployment, incremental updates for the current season, SHAP explanations
on the predictor page, and a playoff-series simulator.
