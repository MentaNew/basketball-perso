import plotly.express as px
import streamlit as st

from app.utils import (
    ACCENT,
    apply_dark_theme,
    cached_team_list,
    metric_card,
    model_available,
    season_label,
    style_fig,
    win_gauge,
)

st.set_page_config(page_title="Game Predictor", page_icon="🏀", layout="wide")
apply_dark_theme()
st.title("Game Predictor")

if not model_available("game_predictor.pkl"):
    st.info("Train the model first: `python -m ml.game_predictor train`")
    st.stop()

from ml.game_predictor import (  # noqa: E402
    available_seasons,
    available_teams,
    feature_importance,
    get_model_stats,
    predict_winner,
)

stats = get_model_stats()
seasons = available_seasons()
season = st.select_slider("Team strength as of the end of", seasons, value=seasons[-1],
                          format_func=season_label)
teams = cached_team_list(season)
teams = teams[teams.team_abr.isin(available_teams(season))]
names = dict(zip(teams.full_name, teams.team_abr, strict=True))

c1, c2, c3 = st.columns([2, 2, 1])
home = c1.selectbox("Home team", list(names), index=0)
away = c2.selectbox("Away team", list(names), index=min(1, len(names) - 1))
playoff = c3.toggle("Playoff game")
neutral = c3.toggle("Neutral site", help="No home-court advantage (games abroad, 2020 bubble)")

if home == away:
    st.warning("Pick two different teams.")
    st.stop()

res = predict_winner(names[home], names[away], season, playoff, neutral)
g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(win_gauge(res["home_win_prob"], home if neutral else f"{home} (home)"),
                    width="stretch")
with g2:
    st.plotly_chart(win_gauge(res["away_win_prob"], away if neutral else f"{away} (away)"),
                    width="stretch")
m = st.columns(3)
with m[0]:
    winner = home if res["predicted_winner"] == names[home] else away
    metric_card("Predicted winner", winner)
with m[1]:
    metric_card("Confidence", res["confidence"])
with m[2]:
    metric_card("Elo-only home win prob", f"{res['elo_home_prob']:.1%}")

with st.expander("How good is the model?"):
    lo, hi = stats["test_seasons"]
    st.markdown(f"Scored on held-out seasons **{season_label(lo)} – {season_label(hi)}** "
                "against two baselines. Features use only information available before "
                "tip-off (Elo, season-to-date and last-10 form, last season, rest days).")
    st.dataframe(stats["test_report"].style.format(precision=4), hide_index=True)
    imp = feature_importance().head(12).reset_index()
    imp.columns = ["feature", "importance"]
    fig = px.bar(imp, x="importance", y="feature", orientation="h", title="Feature importance")
    fig.update_traces(marker_color=ACCENT)
    fig.update_layout(yaxis=dict(autorange="reversed", title=None))
    st.plotly_chart(style_fig(fig), width="stretch")
