"""Shared feature engineering for all models (training and inference use the same code)."""
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from db.connection import query_df
from etl.parsers import search_name
from etl.teams import franchise_of

# ---------------------------------------------------------------- loaders


def load_player_seasons(min_games: int = 0, primary_only: bool = True) -> pd.DataFrame:
    """One row per player-season (the TOT row for traded players) with identity columns."""
    return query_df(
        f"""SELECT p.bref_id, p.name, p.search_name, p.birth_year, s.*
            FROM player_seasons s JOIN players p ON p.id = s.player_id
            WHERE s.g >= :min_games {'AND s.is_primary' if primary_only else ''}
            ORDER BY p.bref_id, s.season""",
        {"min_games": min_games},
    )


def load_team_seasons() -> pd.DataFrame:
    return query_df("SELECT * FROM team_seasons ORDER BY season, team_abr")


def load_games(types: tuple[str, ...] = ("regular", "play_in", "playoff")) -> pd.DataFrame:
    df = query_df(
        "SELECT * FROM games WHERE game_type = ANY(:t) ORDER BY game_date, game_id",
        {"t": list(types)},
    )
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def resolve_player(df: pd.DataFrame, player: str) -> str | None:
    """Accept a bref_id or a (case/accent-insensitive) name; return the bref_id.
    Name collisions resolve to the player with the most seasons in `df`."""
    if player in set(df["bref_id"]):
        return player
    hits = df[df["search_name"] == search_name(player)]
    if hits.empty:
        return None
    return hits["bref_id"].value_counts().idxmax()


# ---------------------------------------------------------------- game features
#
# Everything is computed from games strictly before tip-off (the v1 model used the
# same season's *final* standings, leaking the result; blueprint §7.2).

ELO_BASE, ELO_K, ELO_HCA, ELO_CARRY = 1505.0, 20.0, 100.0, 0.75
PRIOR_GAMES = 10          # season-to-date stats are shrunk toward last season with this weight
REST_CAP = 7


def elo_win_prob(elo_home: float, elo_away: float, hca: float = ELO_HCA) -> float:
    return 1.0 / (1.0 + 10 ** (-(elo_home + hca - elo_away) / 400.0))


def elo_update(elo_home: float, elo_away: float, home_margin: int,
               hca: float = ELO_HCA) -> tuple[float, float]:
    """FiveThirtyEight-style update with a margin-of-victory multiplier."""
    p_home = elo_win_prob(elo_home, elo_away, hca)
    home_won = home_margin > 0
    winner_diff = (elo_home + hca - elo_away) * (1 if home_won else -1)
    mov_mult = (abs(home_margin) + 3) ** 0.8 / (7.5 + 0.006 * winner_diff)
    delta = ELO_K * mov_mult * ((1.0 if home_won else 0.0) - p_home)
    return elo_home + delta, elo_away - delta


@dataclass
class TeamState:
    elo: float = ELO_BASE
    season: int | None = None
    gp: int = 0
    wins: int = 0
    net: float = 0.0                       # cumulative point differential this season
    last10: deque = field(default_factory=lambda: deque(maxlen=10))  # (win, margin)
    last_date: pd.Timestamp | None = None
    prev_w_pct: float = 0.5
    prev_net: float = 0.0

    def start_season(self, season: int):
        if self.season is not None and self.gp:
            self.prev_w_pct, self.prev_net = self.wins / self.gp, self.net / self.gp
        if self.season is not None:
            self.elo = ELO_CARRY * self.elo + (1 - ELO_CARRY) * ELO_BASE
        self.season, self.gp, self.wins, self.net = season, 0, 0, 0.0
        self.last10.clear()
        self.last_date = None

    def features(self, date: pd.Timestamp | None) -> dict:
        n, k = self.gp, PRIOR_GAMES
        rest = REST_CAP if self.last_date is None or date is None else min(
            (date - self.last_date).days, REST_CAP)
        l10 = list(self.last10)
        return {
            "elo": self.elo,
            "w_pct": (self.wins + k * self.prev_w_pct) / (n + k),
            "net": (self.net + k * self.prev_net) / (n + k),
            "l10_w_pct": np.mean([w for w, _ in l10]) if l10 else self.prev_w_pct,
            "l10_net": np.mean([m for _, m in l10]) if l10 else self.prev_net,
            "prev_w_pct": self.prev_w_pct,
            "prev_net": self.prev_net,
            "gp": n,
            "rest": rest,
            "b2b": int(rest == 1),
        }

    def record(self, date: pd.Timestamp, margin: int):
        self.gp += 1
        self.wins += margin > 0
        self.net += margin
        self.last10.append((int(margin > 0), margin))
        self.last_date = date


TEAM_FEATURES = ["elo", "w_pct", "net", "l10_w_pct", "l10_net", "prev_w_pct", "prev_net",
                 "gp", "rest", "b2b"]
GAME_FEATURES = (
    [f"home_{f}" for f in TEAM_FEATURES] + [f"away_{f}" for f in TEAM_FEATURES]
    + ["elo_diff", "net_diff", "w_pct_diff", "elo_prob", "is_playoff"]
)


def matchup_features(home: dict, away: dict, is_playoff: bool) -> dict:
    row = {f"home_{k}": v for k, v in home.items()} | {f"away_{k}": v for k, v in away.items()}
    row["elo_diff"] = home["elo"] - away["elo"]
    row["net_diff"] = home["net"] - away["net"]
    row["w_pct_diff"] = home["w_pct"] - away["w_pct"]
    row["elo_prob"] = elo_win_prob(home["elo"], away["elo"])
    row["is_playoff"] = int(is_playoff)
    return row


def build_game_features(games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Walk games chronologically, emitting pre-game features, then updating state.

    Returns (features DataFrame aligned with `games` + target, end-of-season snapshots
    {(season, team_abr): team feature dict}) for hypothetical matchups at inference time.
    Team state is keyed by franchise, so Seattle's history carries into Oklahoma City.
    """
    states: dict[str, TeamState] = {}
    snapshots: dict[tuple[int, str], dict] = {}
    last_abr: dict[str, str] = {}
    rows = []
    current_season = None

    def snapshot_season(season):
        for fr, st in states.items():
            if st.season == season:
                snapshots[(season, last_abr[fr])] = st.features(None) | {"rest": 2, "b2b": 0}

    for g in games.itertuples(index=False):
        if g.season != current_season:
            if current_season is not None:
                snapshot_season(current_season)
            current_season = g.season
        hf, af = franchise_of(g.home_team), franchise_of(g.away_team)
        for fr, abr in ((hf, g.home_team), (af, g.away_team)):
            st = states.setdefault(fr, TeamState())
            if st.season != g.season:
                st.start_season(g.season)
            last_abr[fr] = abr
        h, a = states[hf], states[af]
        rows.append(matchup_features(h.features(g.game_date), a.features(g.game_date),
                                     g.game_type == "playoff"))
        margin = int(g.home_score - g.away_score)
        h.elo, a.elo = elo_update(h.elo, a.elo, margin)
        h.record(g.game_date, margin)
        a.record(g.game_date, -margin)
    if current_season is not None:
        snapshot_season(current_season)

    feats = pd.DataFrame(rows, index=games.index)
    feats["target"] = games["home_win"].astype(int).values
    return feats, snapshots


# ---------------------------------------------------------------- similarity features

SIMILARITY_COLS = ["pts", "reb", "ast", "stl", "blk", "tov", "fg_pct", "fg3_pct", "ft_pct",
                   "ts_pct", "usg_pct", "per", "bpm", "fg3a_rate"]
RAW_DISPLAY_COLS = ["pts", "reb", "ast", "stl", "blk", "fg3_pct", "ts_pct", "per", "bpm"]


def build_similarity_features(min_games: int = 41) -> pd.DataFrame:
    """Per-season z-scores (era adjustment: pace and the 3PT revolution) of SIMILARITY_COLS.
    z-columns are suffixed `_z`; raw columns are kept for display."""
    df = load_player_seasons(min_games=min_games)
    df = df.dropna(subset=["pts", "per"]).reset_index(drop=True)
    grp = df.groupby("season")
    for c in SIMILARITY_COLS:
        z = (df[c] - grp[c].transform("mean")) / (grp[c].transform("std") + 1e-8)
        df[f"{c}_z"] = z.fillna(0.0)  # e.g. fg3_pct for players with no attempts
    return df


def z_cols() -> list[str]:
    return [f"{c}_z" for c in SIMILARITY_COLS]


def career_profiles(season_df: pd.DataFrame) -> pd.DataFrame:
    """Games-weighted career average of the per-season z-scores (one row per player)."""
    zc = z_cols()
    w = season_df["g"]
    agg = season_df[zc].mul(w, axis=0).groupby(season_df["bref_id"]).sum().div(
        w.groupby(season_df["bref_id"]).sum(), axis=0)
    meta = season_df.groupby("bref_id").agg(
        name=("name", "first"), search_name=("search_name", "first"),
        first_season=("season", "min"), last_season=("season", "max"),
        seasons=("season", "count"),
        **{c: (c, "mean") for c in RAW_DISPLAY_COLS},
    )
    return meta.join(agg).reset_index()


# ---------------------------------------------------------------- career sequences (LSTM)

LSTM_STAT_COLS = ["pts", "reb", "ast", "stl", "blk", "fg_pct"]
LSTM_INPUT_COLS = LSTM_STAT_COLS + ["mp", "g", "age", "gap"]
SEQ_IN = 4           # seasons of history
MAX_GAP = 2          # allow one missed season inside the input window


def career_table(min_games: int = 20) -> pd.DataFrame:
    df = load_player_seasons(min_games=min_games)
    df = df.dropna(subset=LSTM_STAT_COLS + ["mp", "age"]).sort_values(["bref_id", "season"])
    df["gap"] = df.groupby("bref_id")["season"].diff().fillna(1).clip(upper=3)
    return df.reset_index(drop=True)


def build_career_sequences(df: pd.DataFrame) -> dict:
    """Sliding windows: SEQ_IN seasons in (gaps ≤ MAX_GAP, gap is a feature) -> the
    immediately following season out.  Returned unnormalised; the model normalises."""
    X, y, meta = [], [], []
    for bref_id, grp in df.groupby("bref_id", sort=False):
        vals = grp[LSTM_INPUT_COLS].to_numpy(np.float32)
        targ = grp[LSTM_STAT_COLS].to_numpy(np.float32)
        seasons = grp["season"].to_numpy()
        gaps = grp["gap"].to_numpy()
        for end in range(SEQ_IN, len(grp)):
            window_gaps = gaps[end - SEQ_IN + 1:end]
            if gaps[end] != 1 or (window_gaps > MAX_GAP).any():
                continue
            X.append(vals[end - SEQ_IN:end])
            y.append(targ[end])
            meta.append((bref_id, grp["name"].iat[0], int(seasons[end]), float(grp["age"].iat[end])))
    return {
        "X": np.stack(X) if X else np.zeros((0, SEQ_IN, len(LSTM_INPUT_COLS)), np.float32),
        "y": np.stack(y) if y else np.zeros((0, len(LSTM_STAT_COLS)), np.float32),
        "meta": pd.DataFrame(meta, columns=["bref_id", "name", "target_season", "target_age"]),
    }
