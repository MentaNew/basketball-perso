"""Player similarity and archetypes.

Pipeline (features from ml.features: per-season z-scores, so eras compare fairly):

    14 z-scores --PCA(10)--> denoised space --+--> KMeans(9)      -> clusters -> archetype names
                                              +--> cosine similarity -> "who plays like X?"
                                              +--> UMAP(2)        -> the 2-D map in the app

Clustering and similarity use the PCA space, never the UMAP coordinates: UMAP distorts
distances to make a readable picture.

Usage: python -m ml.similarity train
"""
import argparse
from functools import cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import umap
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ml.features import (
    RAW_DISPLAY_COLS,
    SIMILARITY_COLS,
    build_similarity_features,
    career_profiles,
    resolve_player,
    z_cols,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "similarity.pkl"
N_PCA, N_CLUSTERS = 10, 9
SAME_ERA_SEASONS = 5  # cross_era=False keeps players within this many seasons

# Archetype templates: the z-score profile each archetype is expected to have. KMeans numbers
# its clusters arbitrarily, so each cluster gets the name of the template its centroid
# resembles most (one name per cluster). There are more templates than clusters; the
# worst-fitting ones simply go unused.
ARCHETYPE_TEMPLATES = {
    "Franchise Star":     {"pts": 1.8, "usg_pct": 1.6, "per": 1.9, "bpm": 1.7, "ast": 0.6, "reb": 0.5, "ts_pct": 0.6},
    "Traditional Big":   {"reb": 1.5, "blk": 1.2, "fg_pct": 1.2, "fg3a_rate": -1.2, "fg3_pct": -0.8, "ast": -0.5, "ft_pct": -0.6},
    "Modern Stretch Big": {"reb": 1.0, "blk": 0.8, "fg3a_rate": 0.6, "fg3_pct": 0.5, "ft_pct": 0.3},
    "Two-Way Anchor":     {"reb": 1.2, "blk": 1.2, "stl": 0.8, "bpm": 1.3, "per": 0.8, "usg_pct": -0.2},
    "Floor General":      {"ast": 1.8, "tov": 1.2, "stl": 0.8, "bpm": 0.8, "per": 0.5, "reb": -0.4, "blk": -0.6},
    "Pass-First Guard":   {"ast": 1.2, "pts": -0.4, "usg_pct": -0.3, "reb": -0.6, "blk": -0.7, "fg3a_rate": 0.3},
    "Scoring Guard":      {"pts": 1.3, "usg_pct": 1.3, "ast": 0.8, "tov": 0.8, "ft_pct": 0.6, "reb": -0.5, "blk": -0.5},
    "Perimeter Scorer":   {"pts": 1.2, "usg_pct": 1.0, "fg3a_rate": 0.8, "fg3_pct": 0.5, "ts_pct": 0.5, "reb": -0.2},
    "3-and-D Wing":       {"fg3a_rate": 1.0, "fg3_pct": 0.8, "stl": 0.4, "usg_pct": -0.6, "ast": -0.3, "reb": -0.3},
    "Athletic Slasher":   {"fg_pct": 0.8, "pts": 0.6, "reb": 0.4, "stl": 0.4, "fg3a_rate": -0.8, "fg3_pct": -0.4},
    "Energy Big":         {"reb": 0.9, "blk": 0.6, "fg_pct": 0.6, "pts": -0.6, "usg_pct": -0.8, "ft_pct": -0.5},
    "Bench Shooter":      {"fg3a_rate": 0.8, "pts": -0.6, "usg_pct": -0.4, "per": -0.6, "bpm": -0.5},
    "Low-Usage Role Player": {"pts": -0.8, "usg_pct": -0.8, "per": -0.8, "bpm": -0.8, "ts_pct": -0.5},
}


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def name_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    """{cluster id: archetype name}, from centroids (clusters x SIMILARITY_COLS mean z-scores).

    Cosine similarity between every centroid and every template, then the Hungarian
    algorithm picks the one-to-one assignment with the highest total similarity.
    """
    names = list(ARCHETYPE_TEMPLATES)
    templates = np.array([[ARCHETYPE_TEMPLATES[n].get(c, 0.0) for c in SIMILARITY_COLS] for n in names])
    similarity = unit_rows(centroids[SIMILARITY_COLS].to_numpy()) @ unit_rows(templates).T
    cluster_rows, template_cols = linear_sum_assignment(-similarity)  # it minimises cost
    return {int(centroids.index[r]): names[c] for r, c in zip(cluster_rows, template_cols, strict=True)}


def train() -> dict:
    seasons = build_similarity_features(min_games=41)
    print(f"Player-seasons: {len(seasons):,} ({seasons.season.min()}–{seasons.season.max()})")

    pca = PCA(n_components=N_PCA, random_state=42).fit(seasons[z_cols()].to_numpy())
    season_vectors = pca.transform(seasons[z_cols()].to_numpy())
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=42).fit(season_vectors)
    seasons["cluster"] = kmeans.labels_
    centroids = seasons.groupby("cluster")[z_cols()].mean()
    centroids.columns = SIMILARITY_COLS
    archetypes = name_clusters(centroids)
    seasons["archetype"] = seasons["cluster"].map(archetypes)

    print("UMAP…")
    reducer = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric="cosine", random_state=42)
    seasons[["umap_x", "umap_y"]] = reducer.fit_transform(season_vectors)

    careers = career_profiles(seasons)
    career_vectors = pca.transform(careers[z_cols()].to_numpy())
    careers["cluster"] = kmeans.predict(career_vectors)
    careers["archetype"] = careers["cluster"].map(archetypes)

    print("\nArchetypes (size · strongest centroid z-scores):")
    for cluster, name in sorted(archetypes.items()):
        top = centroids.loc[cluster].sort_values(key=abs, ascending=False).head(4)
        profile = ", ".join(f"{stat} {z:+.1f}" for stat, z in top.items())
        print(f"  {cluster} {name:<22} n={int((seasons.cluster == cluster).sum()):>5}  {profile}")

    columns = list(dict.fromkeys(  # dict.fromkeys de-duplicates while keeping order
        ["bref_id", "name", "search_name", "season", "team_abr", "team_list", "pos", "age", "g",
         *RAW_DISPLAY_COLS, *z_cols(), "cluster", "archetype", "umap_x", "umap_y"]))
    bundle = {
        "df": seasons[columns].reset_index(drop=True),
        "X_pca": season_vectors.astype(np.float32),
        "career": careers,
        "career_pca": career_vectors.astype(np.float32),
        "pca": pca, "kmeans": kmeans, "centroids": centroids,
        "archetype_names": archetypes, "feature_cols": SIMILARITY_COLS,
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"\nSaved {MODEL_PATH}")
    return bundle


# ---------------------------------------------------------------- inference

@cache
def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


def cosine_to(vectors: np.ndarray, target: np.ndarray) -> np.ndarray:
    return (vectors @ target) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(target) + 1e-12)


def find_similar(player: str, season: int | None = None, top_k: int = 10,
                 cross_era: bool = True, mode: str = "season") -> pd.DataFrame:
    """The players most similar to `player` (a name or bref_id), never the player themself.

    mode="season": one of their seasons (default: latest) against every player-season;
                   each other player appears once, at their best-matching season.
    mode="career": games-weighted career profiles against each other.
    cross_era=False keeps only players from within SAME_ERA_SEASONS seasons.
    """
    if mode == "career":
        return _similar_careers(player, top_k, cross_era)
    return _similar_seasons(player, season, top_k, cross_era)


def _similar_seasons(player: str, season: int | None, top_k: int, cross_era: bool) -> pd.DataFrame:
    bundle = load_bundle()
    seasons = bundle["df"]
    bref_id = resolve_player(seasons, player)
    if bref_id is None:
        raise KeyError(player)
    own = seasons[seasons.bref_id == bref_id]
    target = own[own.season == (season or own.season.max())]
    if target.empty:
        raise KeyError(f"{player} has no qualifying season {season}")
    i = target.index[0]

    out = seasons.assign(similarity=cosine_to(bundle["X_pca"], bundle["X_pca"][i]))
    out = out[out.bref_id != bref_id]
    if not cross_era:
        out = out[(out.season - seasons.at[i, "season"]).abs() <= SAME_ERA_SEASONS]
    out = out.sort_values("similarity", ascending=False).drop_duplicates("bref_id")
    return out.head(top_k)[["bref_id", "name", "season", "team_abr", "archetype", "similarity",
                            "pts", "reb", "ast"]].reset_index(drop=True)


def _similar_careers(player: str, top_k: int, cross_era: bool) -> pd.DataFrame:
    bundle = load_bundle()
    careers = bundle["career"]
    bref_id = resolve_player(careers, player)
    if bref_id is None:
        raise KeyError(player)
    i = careers.index[careers.bref_id == bref_id][0]

    out = careers.assign(similarity=cosine_to(bundle["career_pca"], bundle["career_pca"][i]))
    out = out[out.bref_id != bref_id]
    if not cross_era:
        midpoint = (out.first_season + out.last_season) / 2
        own_midpoint = (careers.at[i, "first_season"] + careers.at[i, "last_season"]) / 2
        out = out[(midpoint - own_midpoint).abs() <= SAME_ERA_SEASONS]
    return out.nlargest(top_k, "similarity")[["bref_id", "name", "first_season", "last_season",
                                              "archetype", "similarity", "pts", "reb", "ast"]
                                             ].reset_index(drop=True)


def get_archetypes() -> pd.DataFrame:
    """One row per cluster: archetype name, size, and centroid z-scores."""
    bundle = load_bundle()
    table = bundle["centroids"].copy()
    table.insert(0, "archetype", table.index.map(bundle["archetype_names"]))
    table.insert(1, "n", bundle["df"].cluster.value_counts().reindex(table.index).values)
    return table.reset_index()


def get_player_archetype(player: str, season: int | None = None) -> dict | None:
    """The archetype of one of a player's seasons (default: latest), or None."""
    seasons = load_bundle()["df"]
    bref_id = resolve_player(seasons, player)
    if bref_id is None:
        return None
    own = seasons[seasons.bref_id == bref_id]
    row = own[own.season == (season or own.season.max())]
    if row.empty:
        return None
    row = row.iloc[0]
    return {"name": row["name"], "season": int(row.season), "archetype": row.archetype,
            "cluster": int(row.cluster)}


def get_umap_plot_data() -> pd.DataFrame:
    """2-D map coordinates with raw (not z-scored) stats for tooltips."""
    return load_bundle()["df"][["bref_id", "name", "season", "team_abr", "archetype",
                                "umap_x", "umap_y", *RAW_DISPLAY_COLS]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the player similarity model.")
    parser.add_argument("command", choices=["train"])
    parser.parse_args()
    train()
