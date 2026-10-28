"""Predict each team's regular-season wins before the season starts (roadmap Track A).

One row per (season, team). Target: `wins82` = win% x 82, so lockout and COVID seasons are
on the same scale. The table also holds one row per franchise for the *next* season
(target unknown): that's what `predict()` projects.

**The rule for every feature:** it must be known before opening night of `season`. Build it
from earlier seasons only (see `add_last_season`: shift by one season *within a franchise*).

Model: Ridge on last season's record and the roster's strength (last season's VORP and BPM
of the players on this season's roster). See ML_NEXT_STEPS.md for results and ideas.

Usage:
    python -m ml.season_wins train      # fit, compare with baselines, log, save
    python -m ml.season_wins predict    # projections for the next season
"""
import argparse
from collections.abc import Callable
from functools import cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from db.connection import query_df
from etl.scraper import get_roster
from etl.teams import franchise_of
from ml.experiments import log_run
from ml.features import read_player_seasons

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "season_wins.pkl"

TRAIN_MAX, VAL_MAX = 2021, 2024          # train <= 2021, validation 2022-24, test 2025+
SHORT_SEASONS = {1999, 2012, 2020, 2021}  # not 82 games: kept for training, never evaluated


# ================================================================ data

def read_team_seasons() -> pd.DataFrame:
    """Every completed team-season, plus one row per franchise for the next season.

    Columns: season, team_abr, franchise_id, wins82 (target; NaN for next season), net
    (points scored - allowed per game), srs, vegas_line (NaN before 1994 and where the
    archive has no line), payroll_to_cap (that season's *full-year* payroll; see add_payroll).
    """
    seasons = query_df("""
        SELECT t.season, t.team_abr, t.franchise_id, t.w_pct * 82 AS wins82,
               t.ps_g - t.pa_g AS net, t.srs, w.line AS vegas_line, p.payroll_to_cap
        FROM team_seasons t
        LEFT JOIN win_totals w USING (season, team_abr)
        LEFT JOIN team_payrolls p USING (season, team_abr)""")
    upcoming = query_df("SELECT current_abr AS team_abr, franchise_id FROM franchises")
    upcoming["season"] = seasons["season"].max() + 1
    table = pd.concat([seasons, upcoming], ignore_index=True)
    return table.sort_values(["franchise_id", "season"], ignore_index=True)


# ================================================================ features (A2)
# Each builder takes the table and returns it with new columns. Builders run in order.

def add_last_season(df: pd.DataFrame) -> pd.DataFrame:
    """Worked example. prev_wins82, prev_net, prev_srs: the franchise's previous season.

    shift(1) inside each franchise means a row only ever sees an earlier season (Seattle's
    2008 becomes Oklahoma City's "last season" in 2009). An expansion team's first season
    gets NaN.
    """
    by_franchise = df.groupby("franchise_id")
    for col in ("wins82", "net", "srs"):
        df[f"prev_{col}"] = by_franchise[col].shift(1)
    return df


def add_progression(df: pd.DataFrame) -> pd.DataFrame:
    """`progression` = last season's wins82 minus the season before's (needs add_last_season)."""
    df["progression"] = df["prev_wins82"] - df.groupby("franchise_id")["prev_wins82"].shift(1)
    return df


def season_rosters(upcoming: int) -> pd.DataFrame:
    """(season, franchise_id, bref_id): who is on each team, for every season incl. `upcoming`.

    Completed seasons: each player counts for the team he *started* the season with (first
    entry of team_list), a proxy for the opening-night roster. Upcoming season: today's roster
    pages on basketball-reference (fetched once, then cached).
    shortcut: completed seasons miss players who sat out the whole year (~1% of VORP, mostly
    known before the season); real opening-night rosters would remove that small leak.
    """
    played = read_player_seasons()
    played["franchise_id"] = played["team_list"].str.split(",").str[0].map(franchise_of)
    today = [{"season": upcoming, "franchise_id": franchise_of(team), "bref_id": bref_id}
             for team in query_df("SELECT current_abr FROM franchises")["current_abr"]
             for bref_id in get_roster(upcoming, team)]
    return pd.concat([played[["season", "franchise_id", "bref_id"]], pd.DataFrame(today)], ignore_index=True)


def add_roster_strength(df: pd.DataFrame) -> pd.DataFrame:
    """How good this season's roster was over the two previous seasons (all known in October).
    See `team_strength` for the three features it adds."""
    stats = read_player_seasons()[["bref_id", "season", "vorp", "bpm", "mp_total", "age"]]
    last = stats.assign(season=stats["season"] + 1)                       # Y-1 describes season Y
    before = stats.drop(columns="age").assign(season=stats["season"] + 2)  # Y-2
    roster = (season_rosters(int(df["season"].max()))
              .merge(last, on=["bref_id", "season"], how="left")
              .merge(before, on=["bref_id", "season"], how="left", suffixes=("", "_2")))
    return df.merge(team_strength(roster), on=["season", "franchise_id"], how="left")


def team_strength(roster: pd.DataFrame) -> pd.DataFrame:
    """Per (season, franchise_id), from one row per rostered player with his stats one season
    back (vorp, bpm, mp_total, age) and two seasons back (vorp_2, bpm_2, mp_total_2):

    roster_vorp: sum of VORP, Marcel-style: last season weighs 2, the one before 1
                 (backtest 2005-24: 6.85 MAE vs 7.02 with last season only).
    roster_bpm:  BPM over the same two seasons, weighted by minutes x (2 or 1); rookies weigh 0
                 (backtest: 6.81 vs 6.84 with last season only).
    roster_age:  age this season, weighted by last season's minutes.
    """
    marcel = (2 * roster["vorp"].fillna(0) + roster["vorp_2"]) / 3
    vorp = roster["vorp"].where(roster["vorp_2"].isna(), marcel)
    minutes, minutes_2 = roster["mp_total"].fillna(0), roster["mp_total_2"].fillna(0)
    parts = roster[["season", "franchise_id"]].assign(
        vorp=vorp,
        bpm_x_w=2 * minutes * roster["bpm"].fillna(0) + minutes_2 * roster["bpm_2"].fillna(0),
        bpm_w=2 * minutes + minutes_2,
        age_x_min=(roster["age"] + 1) * minutes,
        min=minutes)
    team = parts.groupby(["season", "franchise_id"]).sum(min_count=1).reset_index()
    return pd.DataFrame({"season": team["season"], "franchise_id": team["franchise_id"],
                         "roster_vorp": team["vorp"], "roster_bpm": team["bpm_x_w"] / team["bpm_w"],
                         "roster_age": team["age_x_min"] / team["min"]})


def add_payroll(df: pd.DataFrame) -> pd.DataFrame:
    """Last season's payroll / cap (this season's full-year payroll would leak mid-season trades).
    Not in FEATURES: it added only noise on validation (6.67 -> 6.64 MAE on 90 rows)."""
    df["prev_payroll_to_cap"] = df.groupby("franchise_id")["payroll_to_cap"].shift(1)
    return df


FEATURE_BUILDERS: list[Callable[[pd.DataFrame], pd.DataFrame]] = [
    add_last_season, add_progression, add_roster_strength, add_payroll]
FEATURES = ["prev_wins82", "prev_net", "roster_vorp", "roster_bpm", "roster_age"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    for builder in FEATURE_BUILDERS:
        df = builder(df)
    return df


# ================================================================ baselines (A1)
# Each returns a Series of predicted wins82, aligned with `df`.

def baseline_always_41(df: pd.DataFrame) -> pd.Series:
    """Worked example: every team is average."""
    return pd.Series(41.0, index=df.index)


def baseline_last_season(df: pd.DataFrame) -> pd.Series:
    """Same as last season."""
    return df["prev_wins82"]


def baseline_regressed(df: pd.DataFrame, keep: float = 0.7) -> pd.Series:
    """Last season pulled toward 41: 41 + keep * (prev_wins82 - 41). keep=0.7: MAE 7.66 (1990-2026)."""
    return 41 + keep * (df["prev_wins82"] - 41)


def baseline_vegas(df: pd.DataFrame) -> pd.Series:
    """The preseason betting line: the benchmark to chase (MAE 6.67 over 1994-2026)."""
    return df["vegas_line"]


BASELINES = {"always_41": baseline_always_41, "last_season": baseline_last_season,
             "regressed": baseline_regressed, "vegas": baseline_vegas}


# ================================================================ evaluation

def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Completed seasons by period. The test seasons are for the final report only."""
    known = df[df["wins82"].notna()]
    return {"train": known[known["season"] <= TRAIN_MAX],
            "val": known[(known["season"] > TRAIN_MAX) & (known["season"] <= VAL_MAX)],
            "test": known[known["season"] > VAL_MAX]}


def evaluate(df: pd.DataFrame, predictions: dict[str, pd.Series]) -> pd.DataFrame:
    """MAE and RMSE (in wins) of each prediction, per split.

    Every model is scored on the *same rows*: those where all predictions exist, from full
    82-game seasons. Otherwise Vegas (1994+) and a model (1981+) would be graded on
    different seasons. `n` says how many rows were compared.
    """
    rows = []
    for name, part in split(df).items():
        common = part[~part["season"].isin(SHORT_SEASONS)]
        common = common.loc[pd.concat([p.reindex(common.index) for p in predictions.values()], axis=1)
                            .notna().all(axis=1)]
        for model, p in predictions.items():
            error = p.reindex(common.index) - common["wins82"]
            rows.append({"split": name, "model": model, "n": len(common),
                         "mae": error.abs().mean(), "rmse": np.sqrt((error ** 2).mean())})
    return pd.DataFrame(rows)


def backtest(table: pd.DataFrame, features: list[str], seasons=range(2005, 2025)) -> float:
    """Rolling-origin MAE: for each season, train on all earlier seasons, predict it. ~500
    predictions instead of the ~90 in `val`, so use this to decide whether a feature helps.
    Scored on full seasons that have a Vegas line (the same rows Vegas is judged on)."""
    errors = []
    for season in seasons:
        if season in SHORT_SEASONS:
            continue
        fit = table[(table["season"] < season) & table["wins82"].notna()].dropna(subset=features)
        test = table[(table["season"] == season) & table["vegas_line"].notna()]
        test = test.dropna(subset=[*features, "wins82"])
        model = make_model().fit(fit[features], fit["wins82"])
        errors.extend(np.abs(model.predict(test[features]) - test["wins82"]))
    return float(np.mean(errors))


# ================================================================ model

def make_model():
    """Ridge: with ~900 training rows it beat gradient boosting on validation (6.67 vs 6.73 MAE)."""
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def train() -> pd.DataFrame:
    table = build_features(read_team_seasons())
    parts = split(table)
    fit_rows = parts["train"].dropna(subset=FEATURES)
    model = make_model().fit(fit_rows[FEATURES], fit_rows["wins82"])

    predictions = {"model": pd.Series(np.nan, index=table.index)}
    usable = table[FEATURES].notna().all(axis=1)
    predictions["model"][usable] = model.predict(table.loc[usable, FEATURES])
    for name, baseline in BASELINES.items():
        predictions[name] = baseline(table)

    report = evaluate(table, predictions)
    print(report.pivot(index="model", columns="split", values="mae").round(2).to_string())

    # ship a model fit on every completed season; the report above stays the honest estimate
    all_rows = table[table["wins82"].notna()].dropna(subset=FEATURES)
    final = make_model().fit(all_rows[FEATURES], all_rows["wins82"])
    val_mae = report.query("split == 'val' and model == 'model'")["mae"].iat[0]
    log_run("season_wins", metrics={"val_mae": val_mae}, params={"features": FEATURES})
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES, "report": report}, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")
    return report


@cache
def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


def predict() -> pd.DataFrame:
    """Projected wins for every franchise in the next season (the rows with no result yet)."""
    bundle = load_bundle()
    table = build_features(read_team_seasons())
    upcoming = table[table["wins82"].isna()].dropna(subset=bundle["features"])
    return (upcoming.assign(projected_wins=bundle["model"].predict(upcoming[bundle["features"]]))
            [["season", "team_abr", "projected_wins"]]
            .sort_values("projected_wins", ascending=False, ignore_index=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Season win-total model (roadmap Track A).")
    parser.add_argument("command", choices=["train", "predict"])
    if parser.parse_args().command == "train":
        train()
    else:
        print(predict().round(1).to_string(index=False))
