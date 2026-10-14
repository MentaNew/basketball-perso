"""Season-wins architecture (roadmap Track A) on a toy table: no database needed.

Tests marked xfail(strict=True) are your TODO list: they define what each stub must do. When
you implement one, pytest reports "XPASS(strict)" as a failure; that's your cue to delete
the marker so the test guards your code from then on.
"""
import numpy as np
import pandas as pd
import pytest

from ml import season_wins as SW

todo = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="TODO: implement the stub")


def toy_table() -> pd.DataFrame:
    """One franchise relocating (SEA -> OKC), one expansion team, plus next-season rows."""
    rows = [
        # season, team, franchise, wins82, net, srs, vegas_line
        (2007, "SEA", "OKC", 31.0, -3.0, -3.0, 35.5),
        (2008, "SEA", "OKC", 20.0, -8.0, -8.0, 29.5),
        (2009, "OKC", "OKC", 23.0, -6.0, -6.0, 26.5),
        (2010, "OKC", "OKC", 50.0, 3.0, 3.0, np.nan),
        (2009, "XPN", "XPN", 40.0, 0.0, 0.0, 38.0),   # expansion team: no previous season
        (2010, "XPN", "XPN", 44.0, 1.0, 1.0, 41.5),
        (2011, "OKC", "OKC", np.nan, np.nan, np.nan, np.nan),   # next season: unknown result
        (2011, "XPN", "XPN", np.nan, np.nan, np.nan, np.nan),
    ]
    columns = ["season", "team_abr", "franchise_id", "wins82", "net", "srs", "vegas_line"]
    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values(["franchise_id", "season"], ignore_index=True)


def by_key(df: pd.DataFrame, col: str) -> dict:
    return {(r.season, r.team_abr): getattr(r, col) for r in df.itertuples()}


# ---------------------------------------------------------------- architecture (passing)

def test_last_season_uses_only_the_previous_season_of_the_franchise():
    prev = by_key(SW.add_last_season(toy_table()), "prev_wins82")
    assert prev[(2009, "OKC")] == 20.0        # Seattle's 2008 carries into Oklahoma City
    assert prev[(2011, "OKC")] == 50.0        # next season sees the last completed one
    assert np.isnan(prev[(2009, "XPN")])      # expansion team: nothing to look back on


def test_split_keeps_unknown_seasons_out_and_respects_boundaries(monkeypatch):
    monkeypatch.setattr(SW, "TRAIN_MAX", 2008)
    monkeypatch.setattr(SW, "VAL_MAX", 2009)
    parts = SW.split(toy_table())
    assert set(parts["train"]["season"]) == {2007, 2008}
    assert set(parts["val"]["season"]) == {2009}
    assert set(parts["test"]["season"]) == {2010}
    assert all(p["wins82"].notna().all() for p in parts.values())


def test_evaluate_compares_models_on_the_same_rows(monkeypatch):
    monkeypatch.setattr(SW, "TRAIN_MAX", 2010)
    df = toy_table()
    report = SW.evaluate(df, {"always_41": SW.baseline_always_41(df), "line": df["vegas_line"]})
    train = report[report["split"] == "train"].set_index("model")
    assert (train["n"] == 5).all()   # the 2010 OKC row has no line, so neither model is scored on it
    assert train.loc["always_41", "mae"] == pytest.approx(np.mean([10, 21, 18, 1, 3]))


def test_evaluate_skips_short_seasons(monkeypatch):
    monkeypatch.setattr(SW, "TRAIN_MAX", 2010)
    monkeypatch.setattr(SW, "SHORT_SEASONS", {2009})
    df = toy_table()
    report = SW.evaluate(df, {"always_41": SW.baseline_always_41(df)})
    assert report.query("split == 'train'")["n"].iat[0] == 4   # 6 known rows minus two in 2009


# ---------------------------------------------------------------- your TODOs (A1, A2)

@todo
def test_baseline_last_season():
    df = SW.add_last_season(toy_table())
    assert by_key(df.assign(p=SW.baseline_last_season(df)), "p")[(2009, "OKC")] == 20.0


@todo
def test_baseline_regressed():
    df = SW.add_last_season(toy_table())
    p = by_key(df.assign(p=SW.baseline_regressed(df, keep=0.5)), "p")
    assert p[(2009, "OKC")] == pytest.approx(41 + 0.5 * (20 - 41))


@todo
def test_baseline_vegas():
    df = toy_table()
    assert by_key(df.assign(p=SW.baseline_vegas(df)), "p")[(2008, "SEA")] == 29.5


@todo
def test_progression():
    df = SW.add_progression(SW.add_last_season(toy_table()))
    assert by_key(df, "progression")[(2009, "OKC")] == pytest.approx(20.0 - 31.0)
