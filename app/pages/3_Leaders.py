import streamlit as st

from app.utils import (
    LEADER_STATS,
    PCT_MIN_ATTEMPTS,
    apply_dark_theme,
    cached_leaders,
    cached_season_range,
    season_label,
    stat_bar,
)

st.set_page_config(page_title="Leaders", page_icon="🏀", layout="wide")
apply_dark_theme()
st.title("Season Leaders")

lo, hi = cached_season_range()
c1, c2, c3 = st.columns([2, 2, 1])
season = c1.select_slider("Season", list(range(lo, hi + 1)), value=hi, format_func=season_label)
stat = c2.selectbox("Stat", list(LEADER_STATS), format_func=LEADER_STATS.get)
n = c3.number_input("Top N", 5, 50, 15)

leaders = cached_leaders(season, stat, int(n))
leaders["player"] = leaders["name"] + " (" + leaders["team_abr"].where(
    leaders["team_abr"] != "TOT", leaders["team_list"]) + ")"
st.plotly_chart(stat_bar(leaders, "value", "player",
                         f"{LEADER_STATS[stat]} — {season_label(season)}"), width="stretch")
note = "Minimum 41 games."
if stat in PCT_MIN_ATTEMPTS:
    col, m = PCT_MIN_ATTEMPTS[stat]
    note += f" Minimum {m:g} {col.upper()} per game."
st.caption(note + " Traded players are ranked on their combined line.")
