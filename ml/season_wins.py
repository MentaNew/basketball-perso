"""Predict each team's regular-season wins before the season starts (roadmap Track A).

One row per (season, team). Target: `wins82` = win% x 82, so lockout and COVID seasons are
on the same scale. The table also holds one row per franchise for the *next* season
(target unknown): that's what `predict()` projects.

**The rule for every feature:** it must be known before opening night of `season`. Build it
from earlier seasons only (see `add_last_season`: shift by one season *within a franchise*).

Status: architecture plus a working skeleton. Everything runs end to end with one feature
group; the `TODO(A1)`/`TODO(A2)`/`TODO(A3)` stubs are yours to fill (see ML_NEXT_STEPS.md).
Their tests in tests/test_season_wins.py are marked xfail(strict): when you implement one,
its test starts passing, pytest flags "XPASS", and you delete the marker.

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
from ml.experiments import log_run

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
    """TODO(A2): `progression` = prev_wins82 - the season before's wins82 (needs add_last_season).
    Expect it to be weak: early analysis found ~0 correlation with the regressed baseline's error."""
    raise NotImplementedError


def add_roster_strength(df: pd.DataFrame) -> pd.DataFrame:
    """TODO(A2): roster strength from player data (`ml.features.read_player_seasons`).

    For each (season Y, team): take the players whose *first* team in Y is this team (the first
    entry of team_list on TOT rows), and sum their Y-1 BPM (also try WS/48 and VORP), weighted
    by their Y-1 minutes (mp_total). Never use season-Y stats or minutes: they aren't known
    before opening night. Rookies have no Y-1 row: count them as replacement level (BPM -2)
    or ignore them, and test both.
    """
    raise NotImplementedError


def add_continuity(df: pd.DataFrame) -> pd.DataFrame:
    """TODO(A2): share of the team's Y-1 minutes played by players still on the roster in Y."""
    raise NotImplementedError


def add_age(df: pd.DataFrame) -> pd.DataFrame:
    """TODO(A2): minutes-weighted age of the season-Y roster (age in Y, minutes from Y-1)."""
    raise NotImplementedError


def add_payroll(df: pd.DataFrame) -> pd.DataFrame:
    """TODO(A2): financial flexibility. Careful: `payroll_to_cap` is the *full-season* payroll,
    including mid-season trades, so using season Y's value leaks. Start with the lagged value
    (Y-1) and a luxury-tax flag; only use season Y if you can justify it's known in October."""
    raise NotImplementedError


def add_elo(df: pd.DataFrame) -> pd.DataFrame:
    """TODO(A2): the franchise's end-of-previous-season Elo (`ml.features.build_game_features`
    returns end-of-season `snapshots` keyed by (season, team_abr))."""
    raise NotImplementedError


FEATURE_BUILDERS: list[Callable[[pd.DataFrame], pd.DataFrame]] = [add_last_season]  # add yours as they work
FEATURES = ["prev_wins82", "prev_net"]                                              # columns the model sees


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
    """TODO(A1): same as last season (prev_wins82)."""
    raise NotImplementedError


def baseline_regressed(df: pd.DataFrame, keep: float = 0.7) -> pd.Series:
    """TODO(A1): last season pulled toward 41: 41 + keep * (prev_wins82 - 41).
    Early analysis: keep=0.7 gave MAE 7.66 over 1990-2026."""
    raise NotImplementedError


def baseline_vegas(df: pd.DataFrame) -> pd.Series:
    """TODO(A1): the preseason betting line (vegas_line). The benchmark to chase: MAE 6.67."""
    raise NotImplementedError


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


# ================================================================ model (A3)

def make_model():
    """TODO(A3): Ridge is the right start for ~900 rows; try others only if they beat it on val."""
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
        try:
            predictions[name] = baseline(table)
        except NotImplementedError:
            print(f"  (baseline '{name}' not implemented yet: TODO(A1))")

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
