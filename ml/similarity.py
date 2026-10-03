"""Player similarity & archetypes: per-season z-scores -> PCA(10) -> KMeans(9); UMAP(2) for viz.

Usage: python -m ml.similarity train
"""
import argparse
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

# Archetype templates: expected sign/size of the per-season z-score profile.  Clusters are
# matched one-to-one to the best-fitting templates (Hungarian assignment on cosine
# similarity), so names follow the centroids rather than arbitrary KMeans ids (§7.6).
# There are more templates than clusters; the unused ones are simply left out.
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


def name_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    """centroids: (n_clusters x SIMILARITY_COLS) mean z-scores -> {cluster: name}."""
    names = list(ARCHETYPE_TEMPLATES)
    T = np.array([[ARCHETYPE_TEMPLATES[n].get(c, 0.0) for c in SIMILARITY_COLS] for n in names])
    C = centroids[SIMILARITY_COLS].to_numpy()
    sim = (C / np.linalg.norm(C, axis=1, keepdims=True)) @ (
        T / np.linalg.norm(T, axis=1, keepdims=True)).T
    rows, cols = linear_sum_assignment(-sim)
    return {int(centroids.index[r]): names[c] for r, c in zip(rows, cols, strict=True)}


def train():
    df = build_similarity_features(min_games=41)
    X = df[z_cols()].to_numpy()
    print(f"Player-seasons: {len(df):,} ({df.season.min()}–{df.season.max()})")

    pca = PCA(n_components=N_PCA, random_state=42).fit(X)
    X_pca = pca.transform(X)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=42).fit(X_pca)
    df["cluster"] = kmeans.labels_

    centroids = df.groupby("cluster")[z_cols()].mean()
    centroids.columns = SIMILARITY_COLS
    archetypes = name_clusters(centroids)
    df["archetype"] = df["cluster"].map(archetypes)

    print("UMAP…")
    reducer = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, metric="cosine",
                        random_state=42)
    emb = reducer.fit_transform(X_pca)
    df["umap_x"], df["umap_y"] = emb[:, 0], emb[:, 1]

    career = career_profiles(df)
    career_pca = pca.transform(career[z_cols()].to_numpy())
    career["cluster"] = kmeans.predict(career_pca)
    career["archetype"] = career["cluster"].map(archetypes)

    print("\nArchetypes (size · top centroid z-scores):")
    for k, name in sorted(archetypes.items()):
        top = centroids.loc[k].sort_values(key=abs, ascending=False).head(4)
        desc = ", ".join(f"{c} {v:+.1f}" for c, v in top.items())
        print(f"  {k} {name:<22} n={int((df.cluster == k).sum()):>5}  {desc}")

    keep = ["bref_id", "name", "search_name", "season", "team_abr", "team_list", "pos", "age", "g",
            *RAW_DISPLAY_COLS, *z_cols(), "cluster", "archetype", "umap_x", "umap_y"]
    keep = list(dict.fromkeys(keep))
    bundle = {
        "df": df[keep].reset_index(drop=True), "X_pca": X_pca.astype(np.float32),
        "career": career, "career_pca": career_pca.astype(np.float32),
        "pca": pca, "kmeans": kmeans, "centroids": centroids,
        "archetype_names": archetypes, "feature_cols": SIMILARITY_COLS,
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"\nSaved {MODEL_PATH}")
    return bundle


# ---------------------------------------------------------------- inference

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def _cosine(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (M @ v) / (np.linalg.norm(M, axis=1) * np.linalg.norm(v) + 1e-12)


def find_similar(player: str, season: int | None = None, top_k: int = 10,
                 cross_era: bool = True, mode: str = "season") -> pd.DataFrame:
    """Most similar players by cosine similarity in PCA space.

    mode="season": compare one season (`season`, default the player's latest) against all
    player-seasons; cross_era=False restricts to ±5 seasons.  mode="career": compare
    games-weighted career profiles.  Never returns the player themself.
    """
    b = _load()
    if mode == "career":
        career = b["career"]
        bref_id = resolve_player(career, player)
        if bref_id is None:
            raise KeyError(player)
        idx = career.index[career.bref_id == bref_id][0]
        sims = _cosine(b["career_pca"], b["career_pca"][idx])
        out = career.assign(similarity=sims)
        out = out[out.bref_id != bref_id]
        if not cross_era:
            mid = (career.at[idx, "first_season"] + career.at[idx, "last_season"]) / 2
            out = out[((out.first_season + out.last_season) / 2 - mid).abs() <= 5]
        return out.nlargest(top_k, "similarity")[
            ["bref_id", "name", "first_season", "last_season", "archetype", "similarity",
             "pts", "reb", "ast"]].reset_index(drop=True)

    df = b["df"]
    bref_id = resolve_player(df, player)
    if bref_id is None:
        raise KeyError(player)
    rows = df[df.bref_id == bref_id]
    target = rows[rows.season == season] if season else rows[rows.season == rows.season.max()]
    if target.empty:
        raise KeyError(f"{player} has no qualifying season {season}")
    idx = target.index[0]
    sims = _cosine(b["X_pca"], b["X_pca"][idx])
    out = df.assign(similarity=sims)
    out = out[out.bref_id != bref_id]
    if not cross_era:
        out = out[(out.season - df.at[idx, "season"]).abs() <= 5]
    # best season per similar player, so one player doesn't fill the list
    out = out.sort_values("similarity", ascending=False).drop_duplicates("bref_id")
    return out.head(top_k)[
        ["bref_id", "name", "season", "team_abr", "archetype", "similarity", "pts", "reb", "ast"]
    ].reset_index(drop=True)


def get_archetypes() -> pd.DataFrame:
    b = _load()
    c = b["centroids"].copy()
    c.insert(0, "archetype", c.index.map(b["archetype_names"]))
    c.insert(1, "n", b["df"].cluster.value_counts().reindex(c.index).values)
    return c.reset_index()


def get_player_archetype(player: str, season: int | None = None) -> dict | None:
    df = _load()["df"]
    bref_id = resolve_player(df, player)
    if bref_id is None:
        return None
    rows = df[df.bref_id == bref_id]
    rows = rows[rows.season == season] if season else rows[rows.season == rows.season.max()]
    if rows.empty:
        return None
    r = rows.iloc[0]
    return {"name": r["name"], "season": int(r.season), "archetype": r.archetype,
            "cluster": int(r.cluster)}


def get_player_archetype_history(player: str) -> pd.DataFrame:
    df = _load()["df"]
    bref_id = resolve_player(df, player)
    return df[df.bref_id == bref_id][["season", "team_abr", "archetype"]]


def get_umap_plot_data() -> pd.DataFrame:
    """Raw (not z-scored) stats for tooltips (§7.6)."""
    return _load()["df"][["bref_id", "name", "season", "team_abr", "archetype", "umap_x", "umap_y",
                          *RAW_DISPLAY_COLS]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.parse_args()
    train()
