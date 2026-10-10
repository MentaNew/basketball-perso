# Phase 3 — Player Similarity & Archetypes

Answer two questions across 46 seasons:
1. **"Who plays like X?"** Nearest neighbours in an era-adjusted stat space.
2. **"What kind of player is X?"** Unsupervised clusters with human-readable archetype names.

Files: `ml/features.py` (similarity section), `ml/similarity.py`, `tests/test_features.py`.

---

## Step 0 — The core problem: eras

Raw stats can't be compared across eras:
- In 1985 a team attempted ~3 threes a game; in 2025 it's ~37.
- Pace and scoring have swung by 15+ points per game.

A 1985 forward with 1.0 3PA/game was an outlier. In 2025 that's a non-shooter. Compare raw
numbers and every modern player looks like a "shooter" and every 1980s player like a "big".

**Fix:** z-score every stat **within its own season**. A player is described by how far they
sit from their contemporaries, not by absolute numbers.

---

## Step 1 — Features (`build_similarity_features`)

```python
df = read_player_seasons(min_games=41)   # primary rows only (TOT for traded players)
df = df.dropna(subset=["pts", "per"])    # need both per-game and advanced stats
for c in SIMILARITY_COLS:
    z = (df[c] - season_mean(c)) / (season_std(c) + 1e-8)
    df[f"{c}_z"] = z.fillna(0.0)
```

`SIMILARITY_COLS` (14):

| Group | Columns |
|---|---|
| Volume | `pts`, `reb`, `ast`, `stl`, `blk`, `tov` |
| Efficiency | `fg_pct`, `fg3_pct`, `ft_pct`, `ts_pct` |
| Role & impact | `usg_pct`, `per`, `bpm` |
| Shot profile | `fg3a_rate` (share of shots from three) |

Design notes:
- `min_games=41` (half a season) keeps out small-sample noise. The result is 14,346
  player-seasons from 1980 to 2026.
- `fg3_pct` is NaN for players with no attempts. Filling the z-score with 0 means "league
  average, no information".
- Raw columns (`RAW_DISPLAY_COLS`) are kept next to the `_z` columns, so tooltips show
  "27.1 PPG" rather than "+2.3σ". v1 showed z-scores in tooltips by mistake (§7.6).

---

## Step 2 — Pipeline

```
14 z-scores ──► PCA(10) ──┬──► KMeans(9) ──► cluster → archetype name
                          ├──► cosine similarity ──► find_similar
                          └──► UMAP(2) ──► 2-D map for the app
```

### 2.1 PCA(10): denoise and decorrelate

`pts`, `usg_pct`, `per` and `tov` are strongly correlated (they all mean "has the ball"). In
raw z-space that axis would count four times in every distance. PCA rotates to uncorrelated
components and keeps 10 of 14. Explained variance is 98.3%, with the first component
alone at 40%: the star-vs-bench axis.

### 2.2 KMeans(9) on the PCA output

`KMeans(n_clusters=9, n_init=20, random_state=42)`. Clustering runs in PCA space, **not** on
UMAP coordinates. UMAP distorts global distances to make a pretty 2-D picture, so clusters
found there reflect the projection, not the data.

### 2.3 UMAP(2): visualisation only

`umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric="cosine", random_state=42)`
on the PCA output. It takes about 1 min and is used **only** for the Archetype Explorer
scatter.

### 2.4 Similarity = cosine in PCA space

Cosine compares the **shape** of a profile (which skills stand out) rather than its
magnitude. A star and a role player with the same "shape" (both rebound-heavy and
pass-light) come out similar, which is the "plays like" intuition.

---

## Step 3 — Naming the clusters (`name_clusters`)

**v1 bug (§7.6):** names were hard-coded by cluster index (`0 → '3-and-D Wing'`), but KMeans
numbers its clusters arbitrarily. Retrain and "3-and-D Wing" could be a cluster of centres.

**Fix: match centroids to templates.**

1. Compute each cluster's centroid: the mean z-score per stat.
2. `ARCHETYPE_TEMPLATES` describes 13 archetypes as expected z-profiles, e.g.
   ```python
   "Traditional Big": {"reb": 1.5, "blk": 1.2, "fg_pct": 1.2, "fg3a_rate": -1.2, ...}
   "3-and-D Wing":    {"fg3a_rate": 1.0, "fg3_pct": 0.8, "stl": 0.4, "usg_pct": -0.6, ...}
   ```
3. Build the cosine-similarity matrix (9 clusters × 13 templates) and solve a **one-to-one
   assignment** with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). Each
   cluster gets the best available unique name. With 13 templates and 9 clusters, the 4
   worst-fitting templates are simply unused.

The names now follow the data. The cluster *ids* changed when the 2025-26 season was added
(Franchise Star moved from 2 to 5), but the *names* stayed with the right profiles, which is
exactly what this step is for. `train` prints each cluster's top centroid z-scores so you can
sanity-check them:

```
  5 Franchise Star         n=  952  ast +2.0, pts +2.0, tov +1.9, bpm +1.8
  6 Traditional Big        n= 1144  fg_pct +1.8, ts_pct +1.2, fg3a_rate -1.1, blk +1.0
  4 Two-Way Anchor         n=  622  blk +2.6, reb +2.3, per +1.8, bpm +1.4
  8 3-and-D Wing           n= 2059  fg3a_rate +1.3, fg3_pct +0.7, tov -0.7, reb -0.6
```

To rename something, edit or add a template; never touch the cluster ids.
`test_archetype_names_follow_centroids` checks that scrambled cluster ids still get the
right names.

---

## Step 4 — Career mode (`career_profiles`)

v1's docstring promised a career average but used the latest season (§7.6). Career mode
now really averages:

```python
career_z = Σ(season_z × games) / Σ games      # games-weighted mean of per-season z-scores
```

This gives one row per player (2,544 players with at least one 41-game season). Rows go
through the **same fitted PCA** and are assigned a cluster with `kmeans.predict`.
Example: Tim Duncan's career neighbours are Elton Brand, Pau Gasol, David Robinson, Patrick
Ewing, Anthony Davis and Hakeem Olajuwon.

---

## Step 5 — The saved bundle (`models/similarity.pkl`, ~3 MB compressed)

```python
{
  "df":          per-season rows: ids, raw stats, z-scores, cluster, archetype, umap_x/y,
  "X_pca":       float32 (n_seasons, 10),
  "career":      per-player rows (career mode), "career_pca": float32 (n_players, 10),
  "pca", "kmeans", "centroids", "archetype_names", "feature_cols",
}
```

---

## Step 6 — Inference API

```python
from ml.similarity import find_similar, get_archetypes, get_player_archetype, \
                          get_umap_plot_data

find_similar("Michael Jordan", season=1991, top_k=10, cross_era=True)
#   Kawhi Leonard 2017, Shai Gilgeous-Alexander 2025, LeBron James 2012, Clyde Drexler 1988…
find_similar("Tim Duncan", mode="career")
find_similar("nikola jokic")       # names are accent- and case-insensitive; bref_ids work too
```

- `season=None` uses the player's latest qualifying season.
- `cross_era=False` keeps only neighbours within ±5 seasons (in career mode, ±5 between
  career midpoints).
- Each similar player appears once, at their single best-matching season, so one player
  can't fill the list.
- Duplicate names (two "Tony Mitchell"s) resolve to the one with the most seasons. Pass a
  `bref_id` to be exact.

---

## Step 7 — Run & verify

```bash
export PYTHONPATH=.
python -m ml.similarity train      # ~1–2 min, mostly UMAP
```

Sanity checks (eyeball them, because there is no single metric for "good clusters"):
- [ ] Known players get sensible labels: Gobert → Traditional Big, Chris Paul → Floor
      General, Jokić and Curry → Franchise Star.
- [ ] No cluster is tiny (< 3%) or giant (> 25%). Current sizes run from 622 to 2,231.
- [ ] Same-era neighbours of a modern stretch big include other stretch bigs.
- [ ] The UMAP map shows coloured regions rather than confetti.

---

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| A label doesn't match its centroid | No template fits that cluster well | Add a template describing it and retrain |
| Results shift after an ETL refresh | KMeans/UMAP are refit on new data | Expected. The seed is fixed but the data changed |
| `KeyError` for a player | Fewer than 41 games in every season, or name typo | Try the `bref_id`, or lower `min_games` |
| UMAP numba warnings on first run | JIT compilation | Harmless; the second run is faster |

---

## What Phase 5 Expects From This

`models/similarity.pkl`, plus `find_similar`, `get_player_archetype`, `get_archetypes` and
`get_umap_plot_data`. The Player Profile and Archetype Explorer pages use only these.
