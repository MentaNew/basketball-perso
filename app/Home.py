import plotly.express as px
import streamlit as st

from app.utils import (
    ACCENT,
    apply_dark_theme,
    cached_counts,
    cached_league_averages,
    cached_league_pace,
    cached_season_range,
    metric_card,
    model_available,
    season_label,
    style_fig,
)

st.set_page_config(page_title="NBA Intelligence Platform", page_icon="🏀", layout="wide")
apply_dark_theme()

first, last = cached_season_range()
st.title("🏀 NBA Intelligence Platform")
st.caption(f"{last - first + 1} seasons of Basketball-Reference data "
           f"({season_label(first)} → {season_label(last)}) · PostgreSQL · "
           "XGBoost · PCA/UMAP/KMeans · LSTM")

counts = cached_counts()
cols = st.columns(4)
with cols[0]:
    metric_card("Seasons", int(counts["seasons"]))
with cols[1]:
    metric_card("Players", f"{int(counts['players']):,}")
with cols[2]:
    metric_card("Games", f"{int(counts['games']):,}")
with cols[3]:
    if model_available("game_predictor.pkl"):
        from ml.game_predictor import get_model_stats
        s = get_model_stats()
        metric_card(f"Game model accuracy ({s['test_seasons'][0]}+)", f"{s['accuracy']:.1%}")
    else:
        metric_card("Game model", "not trained")

st.subheader("How the league changed")
avg = cached_league_averages()
pace = cached_league_pace()

c1, c2 = st.columns(2)
with c1:
    fig = px.line(avg, x="season", y="fg3a_rate", markers=True,
                  title="The three-point revolution: share of shots from three")
    fig.update_traces(line_color=ACCENT)
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None)
    st.plotly_chart(style_fig(fig), width="stretch")
with c2:
    fig = px.line(avg, x="season", y="ts_pct", markers=True, title="True shooting %")
    fig.update_traces(line_color=ACCENT)
    fig.update_layout(yaxis_tickformat=".1%", yaxis_title=None, xaxis_title=None)
    st.plotly_chart(style_fig(fig), width="stretch")

c3, c4 = st.columns(2)
with c3:
    fig = px.line(pace, x="season", y="total_pts", markers=True,
                  title="Combined points per game (regular season)")
    fig.update_traces(line_color=ACCENT)
    fig.update_layout(yaxis_title=None, xaxis_title=None)
    st.plotly_chart(style_fig(fig), width="stretch")
with c4:
    fig = px.line(pace, x="season", y="home_win_rate", markers=True,
                  title="Home-court advantage: home win rate")
    fig.update_traces(line_color=ACCENT)
    fig.add_hline(y=0.5, line_dash="dot", line_color="#6b7280")
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None)
    st.plotly_chart(style_fig(fig), width="stretch")
    st.caption("Neutral-site games (abroad, 2020 bubble) excluded. The edge has shrunk "
               "from ~65% in the 1980s to ~55% today.")

st.markdown("""
**Pages** — *Player Profile*: career arcs, archetype, similar players, next-season projection ·
*Team History*: franchise records across relocations · *Leaders*: any stat, any season ·
*Game Predictor*: win probability for any matchup · *Archetype Explorer*: the
UMAP map of every season's player styles.
""")
