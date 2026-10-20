from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.utils import apply_dark_theme, metric_card, predicted_standings, style_fig

st.set_page_config(page_title="My Predictions", page_icon="🏀", layout="wide")
apply_dark_theme()

sheets = sorted((Path(__file__).resolve().parents[2] / "predictions").glob("*.xlsx"))
if not sheets:
    st.title("My Predictions")
    st.info("No sheet yet: run `python predictions/make_sheet.py`, then fill it in Excel.")
    st.stop()
sheet = sheets[-1]  # the latest season
st.title(f"My {sheet.stem} Predictions")
st.caption(f"From `predictions/{sheet.name}`: edit the yellow column in Excel, save, then refresh this page.")

# Only the typed-in cells are read (columns A-F); ranking is recomputed here, so the page
# doesn't depend on Excel having recalculated the formulas.
raw = pd.read_excel(sheet, sheet_name="My picks", usecols="A:F", nrows=30)
picks = pd.DataFrame({"team": raw.iloc[:, 0], "conference": raw.iloc[:, 2],
                      "last_w": raw.iloc[:, 4].str.split("-").str[0].astype(int),
                      "wins": pd.to_numeric(raw.iloc[:, 5], errors="coerce")})
standings = predicted_standings(picks)

filled, total = len(standings), int(standings["wins"].sum())
cols = st.columns(4)
with cols[0]:
    metric_card("Teams picked", f"{filled} / 30")
with cols[1]:
    metric_card("Total wins", f"{total} / 1230" + (" ✓" if filled == 30 and total == 1230 else ""))
if standings.empty:
    st.info("Nothing picked yet: fill the yellow **My wins** column in Excel and save.")
    st.stop()
best = standings.loc[standings["wins"].idxmax()]
riser = standings.loc[standings["vs_last"].idxmax()]
with cols[2]:
    metric_card("Best record", f"{best.team} ({int(best.wins)}-{int(best.losses)})")
with cols[3]:
    metric_card("Biggest riser", f"{riser.team} ({int(riser.vs_last):+d})")
if filled == 30 and total != 1230:
    st.warning(f"Your 30 teams add up to {total} wins, but a season has exactly 1230 "
               f"(every game has one winner): {abs(total - 1230)} {'too many' if total > 1230 else 'short'}.")

ROW_COLORS = {"Playoffs": "#14532d", "Play-in": "#713f12", "Lottery": "#1f2937"}
east, west = st.columns(2)
for column, conference in ((east, "East"), (west, "West")):
    table = standings[standings["conference"] == conference][
        ["seed", "team", "wins", "losses", "gb", "vs_last", "status"]].rename(columns={
            "seed": "#", "team": "Team", "wins": "W", "losses": "L", "gb": "GB",
            "vs_last": "vs last season", "status": "Status"})
    styled = (table.style
              .apply(lambda row: [f"background-color: {ROW_COLORS[row['Status']]}"] * len(row), axis=1)
              .format({"W": "{:.0f}", "L": "{:.0f}", "GB": "{:.0f}", "vs last season": "{:+.0f}"}))
    with column:
        st.subheader(f"{conference}ern Conference")
        st.dataframe(styled, hide_index=True, width="stretch", height=38 * (len(table) + 1))
st.caption("🟩 seeds 1-6: playoffs · 🟧 7-10: play-in · ⬛ 11-15: lottery. Ties broken alphabetically.")

movers = standings.sort_values("vs_last")
fig = go.Figure(go.Bar(x=movers["vs_last"], y=movers["team"], orientation="h",
                       marker_color=["#22c55e" if v > 0 else "#ef4444" for v in movers["vs_last"]],
                       hovertemplate="%{y}: %{x:+d} wins vs last season<extra></extra>"))
fig.update_layout(title="My predicted change in wins vs last season", xaxis_title=None, yaxis_title=None)
st.plotly_chart(style_fig(fig, height=max(320, 22 * len(movers))), width="stretch")
