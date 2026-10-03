import plotly.graph_objects as go
import streamlit as st

from app.utils import (
    ACCENT,
    apply_dark_theme,
    cached_team_history,
    cached_team_list,
    metric_card,
    style_fig,
)

st.set_page_config(page_title="Team History", page_icon="🏀", layout="wide")
apply_dark_theme()
st.title("Team History")
st.caption("Franchises are tracked across relocations and renames "
           "(e.g. Seattle → Oklahoma City, New Jersey → Brooklyn).")

teams = cached_team_list()
labels = dict(zip(teams.full_name, teams.franchise_id, strict=True))
default = list(labels).index("Boston Celtics") if "Boston Celtics" in labels else 0
name = st.selectbox("Franchise", list(labels), index=default)
hist = cached_team_history(labels[name])

best = hist.loc[hist.w_pct.idxmax()]
cols = st.columns(4)
with cols[0]:
    metric_card("Seasons", len(hist))
with cols[1]:
    metric_card("All-time W–L", f"{hist.w.sum()}–{hist.l.sum()}")
with cols[2]:
    metric_card("Playoff appearances", int(hist.made_playoffs.sum()))
with cols[3]:
    metric_card("Best season", f"{best.season - 1}-{str(best.season)[2:]} ({best.w}–{best.l})")

fig = go.Figure()
fig.add_bar(x=hist.season, y=hist.w, name="Wins", marker_color=ACCENT,
            customdata=hist[["full_name", "l"]], hovertemplate="%{x}: %{y}–%{customdata[1]} "
            "(%{customdata[0]})<extra></extra>")
fig.add_scatter(x=hist[hist.made_playoffs].season, y=hist[hist.made_playoffs].w + 2,
                mode="markers", marker=dict(symbol="star", color="#facc15", size=9),
                name="Made playoffs", hoverinfo="skip")
fig.update_layout(title="Wins per season", xaxis_title=None)
st.plotly_chart(style_fig(fig), width="stretch")

fig = go.Figure()
fig.add_scatter(x=hist.season, y=hist.net_rating, mode="lines+markers", name="Net pts/game",
                line_color=ACCENT)
fig.add_scatter(x=hist.season, y=hist.srs, mode="lines", name="SRS",
                line=dict(color="#9ca3af", dash="dot"))
fig.add_hline(y=0, line_color="#6b7280")
fig.update_layout(title="Point differential and Simple Rating System", xaxis_title=None)
st.plotly_chart(style_fig(fig), width="stretch")

st.dataframe(hist, hide_index=True, width="stretch")
