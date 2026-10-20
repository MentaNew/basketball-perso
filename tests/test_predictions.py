import pandas as pd

from app.utils import predicted_standings


def test_standings_rank_within_conference_break_ties_by_name_and_skip_blanks():
    picks = pd.DataFrame({
        "team": ["Celtics", "Bucks", "Knicks", "Nets", "Lakers", "Jazz"],
        "conference": ["East", "East", "East", "East", "West", "West"],
        "wins": [55, 55, 60, None, 40, 30],
        "last_w": [61, 48, 51, 26, 50, 17],
    })
    s = predicted_standings(picks).set_index("team")
    assert "Nets" not in s.index                                        # not picked yet
    assert s.loc[["Knicks", "Bucks", "Celtics"], "seed"].tolist() == [1, 2, 3]  # tie: B before C
    assert s.loc["Lakers", "seed"] == 1 and s.loc["Jazz", "seed"] == 2  # separate conference
    assert s.loc["Celtics", "gb"] == 5 and s.loc["Celtics", "vs_last"] == -6
    assert s.loc["Jazz", "losses"] == 52


def test_status_by_seed():
    picks = pd.DataFrame({"team": [f"T{i:02d}" for i in range(15)], "conference": "East",
                          "wins": range(60, 45, -1), "last_w": 41})
    status = predicted_standings(picks).set_index("seed")["status"]
    assert (status[6], status[7], status[10], status[11]) == ("Playoffs", "Play-in", "Play-in", "Lottery")
