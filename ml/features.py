"""Feature engineering shared by all three models.

Training and inference import the same functions, so a feature can never be computed one
way when training and another way when predicting.

Sections:
1. Reading from the database
2. Game features: Elo and team form, computed only from games played *before* each game
3. Similarity features: per-season z-scores, so players from different eras compare fairly
4. Career sequences: 4-season windows for the LSTM
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from db.connection import query_df
from etl.parsers import search_name
from etl.teams import franchise_of

# ================================================================ 1. reading from the DB


def read_player_seasons(min_games: int = 0) -> pd.DataFrame:
    """One row per player-season (a traded player's combined TOT row), with identity columns."""
    return query_df(
        """SELECT p.bref_id, p.name, p.search_name, p.birth_year, s.*
           FROM player_seasons s JOIN players p ON p.id = s.player_id
           WHERE s.is_primary AND s.g >= :min_games
           ORDER BY p.bref_id, s.season""",
        {"min_games": min_games},
    )


def read_games(types: tuple[str, ...] = ("regular", "play_in", "playoff")) -> pd.DataFrame:
    """Games in the order they were played (Cup finals excluded by default)."""
    games = query_df("SELECT * FROM games WHERE game_type = ANY(:types) ORDER BY game_date, game_id",
                     {"types": list(types)})
    games["game_date"] = pd.to_datetime(games["game_date"])
    return games


def resolve_player(df: pd.DataFrame, player: str) -> str | None:
    """bref_id for a bref_id or a name (any case, accents optional).
    When two players share a name, picks the one with the most seasons in `df`."""
    if player in set(df["bref_id"]):
        return player
    matches = df.loc[df["search_name"] == search_name(player), "bref_id"]
    return matches.value_counts().idxmax() if not matches.empty else None


# ================================================================ 2. game features
#
# The rule: a feature for game g may only use games that finished before g started.
# v1 used each season's *final* standings, which already contain the game's result.
# Here we walk through the games in date order: read both teams' state (= features),
# then update the state with the result. Leakage is impossible by construction.

ELO_BASE = 1505.0   # rating of an average team
ELO_K = 20.0        # how fast ratings move
ELO_HCA = 100.0     # home-court advantage in Elo points (~64% between equal teams)
ELO_CARRY = 0.75    # between seasons, ratings regress 25% toward the mean
PRIOR_GAMES = 10    # early-season stats are blended with last season as 10 pseudo-games
REST_CAP = 7        # days of rest beyond a week don't matter


def elo_win_prob(elo_home: float, elo_away: float, hca: float = ELO_HCA) -> float:
    return 1.0 / (1.0 + 10 ** (-(elo_home + hca - elo_away) / 400.0))


def elo_update(elo_home: float, elo_away: float, home_margin: int,
               hca: float = ELO_HCA) -> tuple[float, float]:
    """New ratings after a game (FiveThirtyEight-style). Bigger wins move ratings more, but
    wins the favourite was expected to win big move them less. Zero-sum."""
    p_home = elo_win_prob(elo_home, elo_away, hca)
    home_won = home_margin > 0
    winner_edge = (elo_home + hca - elo_away) * (1 if home_won else -1)
    margin_multiplier = (abs(home_margin) + 3) ** 0.8 / (7.5 + 0.006 * winner_edge)
    delta = ELO_K * margin_multiplier * (float(home_won) - p_home)
    return elo_home + delta, elo_away - delta


@dataclass
class TeamState:
    """Everything known about one franchise at a point in time."""
    elo: float = ELO_BASE
    season: int | None = None
    games: int = 0
    wins: int = 0
    point_diff: float = 0.0                       # cumulative, this season
    last10: deque = field(default_factory=lambda: deque(maxlen=10))  # (won, margin) pairs
    last_game: pd.Timestamp | None = None
    prev_w_pct: float = 0.5                       # last season's final values
    prev_net: float = 0.0

    def start_season(self, season: int) -> None:
        if self.season is not None:
            if self.games:
                self.prev_w_pct = self.wins / self.games
                self.prev_net = self.point_diff / self.games
            self.elo = ELO_CARRY * self.elo + (1 - ELO_CARRY) * ELO_BASE
        self.season, self.games, self.wins, self.point_diff = season, 0, 0, 0.0
        self.last10.clear()
        self.last_game = None

    def features(self, today: pd.Timestamp | None) -> dict:
        """Pre-game features. `today=None` means "no specific game" (rest is then the cap)."""
        n, k = self.games, PRIOR_GAMES
        if self.last_game is None or today is None:
            rest = REST_CAP
        else:
            rest = min((today - self.last_game).days, REST_CAP)
        recent = list(self.last10)
        return {
            "elo": self.elo,
            "w_pct": (self.wins + k * self.prev_w_pct) / (n + k),
            "net": (self.point_diff + k * self.prev_net) / (n + k),
            "l10_w_pct": np.mean([won for won, _ in recent]) if recent else self.prev_w_pct,
            "l10_net": np.mean([margin for _, margin in recent]) if recent else self.prev_net,
            "prev_w_pct": self.prev_w_pct,
            "prev_net": self.prev_net,
            "gp": n,
            "rest": rest,
            "b2b": int(rest == 1),
        }

    def record(self, day: pd.Timestamp, margin: int) -> None:
        self.games += 1
        self.wins += margin > 0
        self.point_diff += margin
        self.last10.append((int(margin > 0), margin))
        self.last_game = day


TEAM_FEATURES = ["elo", "w_pct", "net", "l10_w_pct", "l10_net", "prev_w_pct", "prev_net",
                 "gp", "rest", "b2b"]
GAME_FEATURES = (
    [f"home_{f}" for f in TEAM_FEATURES] + [f"away_{f}" for f in TEAM_FEATURES]
    + ["elo_diff", "net_diff", "w_pct_diff", "elo_prob", "is_playoff", "is_neutral"]
)


def matchup_features(home: dict, away: dict, is_playoff: bool, neutral: bool = False) -> dict:
    """One model input row from two teams' feature dicts."""
    row = {f"home_{k}": v for k, v in home.items()} | {f"away_{k}": v for k, v in away.items()}
    row["elo_diff"] = home["elo"] - away["elo"]
    row["net_diff"] = home["net"] - away["net"]
    row["w_pct_diff"] = home["w_pct"] - away["w_pct"]
    row["elo_prob"] = elo_win_prob(home["elo"], away["elo"], hca=0.0 if neutral else ELO_HCA)
    row["is_playoff"] = int(is_playoff)
    row["is_neutral"] = int(neutral)
    return row


def build_game_features(games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Walk the games in date order: emit each game's pre-game features, then update.

    Returns:
      features  - one row per game (aligned with `games`), plus the `target` (home win)
      snapshots - {(season, team_abr): team features after its last game}, used to predict
                  hypothetical matchups ("2016 Warriors vs 1996 Bulls")
    State is kept per franchise, so Seattle's rating carries into Oklahoma City.
    """
    states: dict[str, TeamState] = {}
    current_abr: dict[str, str] = {}  # franchise -> abbreviation it used most recently
    snapshots: dict[tuple[int, str], dict] = {}
    rows = []

    def take_snapshots(season: int) -> None:
        for franchise, state in states.items():
            if state.season == season:
                snapshots[(season, current_abr[franchise])] = state.features(None) | {"rest": 2, "b2b": 0}

    season = None
    for g in games.itertuples(index=False):
        if g.season != season:
            if season is not None:
                take_snapshots(season)
            season = g.season
        home_fr, away_fr = franchise_of(g.home_team), franchise_of(g.away_team)
        for franchise, abr in ((home_fr, g.home_team), (away_fr, g.away_team)):
            state = states.setdefault(franchise, TeamState())
            if state.season != g.season:
                state.start_season(g.season)
            current_abr[franchise] = abr
        home, away = states[home_fr], states[away_fr]

        rows.append(matchup_features(home.features(g.game_date), away.features(g.game_date),
                                     is_playoff=g.game_type == "playoff", neutral=g.neutral))
        margin = int(g.home_score - g.away_score)
        home.elo, away.elo = elo_update(home.elo, away.elo, margin,
                                        hca=0.0 if g.neutral else ELO_HCA)
        home.record(g.game_date, margin)
        away.record(g.game_date, -margin)
    if season is not None:
        take_snapshots(season)

    features = pd.DataFrame(rows, index=games.index)
    features["target"] = games["home_win"].astype(int).values
    return features, snapshots


# ================================================================ 3. similarity features

SIMILARITY_COLS = ["pts", "reb", "ast", "stl", "blk", "tov", "fg_pct", "fg3_pct", "ft_pct",
                   "ts_pct", "usg_pct", "per", "bpm", "fg3a_rate"]
RAW_DISPLAY_COLS = ["pts", "reb", "ast", "stl", "blk", "fg3_pct", "ts_pct", "per", "bpm"]


def z_cols() -> list[str]:
    return [f"{c}_z" for c in SIMILARITY_COLS]


def build_similarity_features(min_games: int = 41) -> pd.DataFrame:
    """Each stat as a z-score *within its own season*.

    In 1985 one three-pointer a game made you a shooter; in 2025 it makes you a non-shooter.
    Comparing players by how far they stood from their contemporaries makes eras comparable.
    Raw columns are kept for display; z-scores get a `_z` suffix.
    """
    df = read_player_seasons(min_games=min_games).dropna(subset=["pts", "per"]).reset_index(drop=True)
    by_season = df.groupby("season")
    for c in SIMILARITY_COLS:
        z = (df[c] - by_season[c].transform("mean")) / (by_season[c].transform("std") + 1e-8)
        df[f"{c}_z"] = z.fillna(0.0)  # e.g. fg3_pct for a player with no attempts: "average"
    return df


def career_profiles(seasons: pd.DataFrame) -> pd.DataFrame:
    """One row per player: the games-weighted average of their per-season z-scores."""
    weights = seasons["g"]
    weighted_sum = seasons[z_cols()].mul(weights, axis=0).groupby(seasons["bref_id"]).sum()
    profile = weighted_sum.div(weights.groupby(seasons["bref_id"]).sum(), axis=0)
    info = seasons.groupby("bref_id").agg(
        name=("name", "first"), search_name=("search_name", "first"),
        first_season=("season", "min"), last_season=("season", "max"),
        seasons=("season", "count"),
        **{c: (c, "mean") for c in RAW_DISPLAY_COLS},
    )
    return info.join(profile).reset_index()


# ================================================================ 4. career sequences (LSTM)

LSTM_STAT_COLS = ["pts", "reb", "ast", "stl", "blk", "fg_pct"]   # what we predict
LSTM_INPUT_COLS = LSTM_STAT_COLS + ["mp", "g", "age", "gap"]       # what the model sees
SEQ_IN = 4    # seasons of history per window
MAX_GAP = 2   # a window may skip at most one season (gap = 2)


def career_table(min_games: int = 20) -> pd.DataFrame:
    """Qualifying player-seasons in career order, with `gap` = seasons since the previous one."""
    df = read_player_seasons(min_games=min_games)
    df = df.dropna(subset=LSTM_STAT_COLS + ["mp", "age"]).sort_values(["bref_id", "season"])
    df["gap"] = df.groupby("bref_id")["season"].diff().fillna(1).clip(upper=3)
    return df.reset_index(drop=True)


def build_career_sequences(careers: pd.DataFrame) -> dict:
    """Sliding windows: SEQ_IN seasons in -> the season right after them out.

    A window is kept only if the target season directly follows it (gap == 1) and no gap
    inside it exceeds MAX_GAP. Returned unnormalised; the model normalises with
    training-set statistics only.
    """
    X, y, meta = [], [], []
    for bref_id, career in careers.groupby("bref_id", sort=False):
        inputs = career[LSTM_INPUT_COLS].to_numpy(np.float32)
        targets = career[LSTM_STAT_COLS].to_numpy(np.float32)
        gaps = career["gap"].to_numpy()
        for end in range(SEQ_IN, len(career)):  # `end` = index of the target season
            if gaps[end] != 1 or (gaps[end - SEQ_IN + 1:end] > MAX_GAP).any():
                continue
            X.append(inputs[end - SEQ_IN:end])
            y.append(targets[end])
            meta.append((bref_id, career["name"].iat[0], int(career["season"].iat[end]),
                         float(career["age"].iat[end])))
    return {  # reshape keeps the right shape even when a career has no valid window
        "X": np.array(X, dtype=np.float32).reshape(-1, SEQ_IN, len(LSTM_INPUT_COLS)),
        "y": np.array(y, dtype=np.float32).reshape(-1, len(LSTM_STAT_COLS)),
        "meta": pd.DataFrame(meta, columns=["bref_id", "name", "target_season", "target_age"]),
    }
