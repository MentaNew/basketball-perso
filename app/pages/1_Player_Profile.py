import pandas as pd
import streamlit as st

from app.utils import (
    apply_dark_theme,
    cached_player_career,
    cached_player_list,
    career_line,
    metric_card,
    model_available,
    season_label,
)

st.set_page_config(page_title="Player Profile", page_icon="🏀", layout="wide")
apply_dark_theme()
st.title("Player Profile")

players = cached_player_list()
default = players.index[players["name"] == "LeBron James"]
choice = st.selectbox("Player", players["label"], index=int(default[0]) if len(default) else 0)
player = players[players["label"] == choice].iloc[0]
career = cached_player_career(player.bref_id)
if career.empty:
    st.warning("No seasons found.")
    st.stop()

cols = st.columns(5)
for col, (label, value) in zip(cols, [
    ("Seasons", len(career)),
    ("Career PPG", f"{(career.pts * career.g).sum() / career.g.sum():.1f}"),
    ("Career RPG", f"{(career.reb * career.g).sum() / career.g.sum():.1f}"),
    ("Career APG", f"{(career.ast * career.g).sum() / career.g.sum():.1f}"),
    ("Total Win Shares", f"{career.ws.sum():.1f}"),
], strict=True):
    with col:
        metric_card(label, value)

if model_available("similarity.pkl"):
    from ml.similarity import get_player_archetype
    arch = get_player_archetype(player.bref_id)
    if arch:
        st.markdown(f'Latest archetype ({season_label(arch["season"])}): '
                    f'<span class="badge">{arch["archetype"]}</span>', unsafe_allow_html=True)

tab_arc, tab_sim, tab_proj, tab_table = st.tabs(
    ["Career arc", "Similar players", "Next-season projection", "Season table"])

with tab_arc:
    c1, c2 = st.columns(2)
    charts = [("pts", "Points per game"), ("reb", "Rebounds per game"), ("ast", "Assists per game"),
              ("ts_pct", "True shooting %"), ("per", "PER"), ("bpm", "Box plus/minus")]
    for i, (col, title) in enumerate(charts):
        with (c1 if i % 2 == 0 else c2):
            st.plotly_chart(career_line(career, col, title), width="stretch")

with tab_sim:
    if not model_available("similarity.pkl"):
        st.info("Train the similarity model: `python -m ml.similarity train`")
    else:
        from ml.similarity import find_similar
        c1, c2, c3 = st.columns([2, 1, 1])
        mode = c1.radio("Compare", ["One season", "Whole career"], horizontal=True)
        cross_era = c2.toggle("Cross-era", value=True, help="Off: only players within ±5 seasons")
        top_k = c3.slider("Results", 5, 25, 10)
        try:
            if mode == "One season":
                qualifying = career[career.g >= 41].season.tolist()
                if not qualifying:
                    raise KeyError("no season with 41+ games")
                season = st.select_slider("Season", qualifying, value=qualifying[-1],
                                          format_func=season_label)
                sim = find_similar(player.bref_id, season=season, top_k=top_k, cross_era=cross_era)
            else:
                sim = find_similar(player.bref_id, top_k=top_k, cross_era=cross_era, mode="career")
            st.dataframe(sim.drop(columns="bref_id"), hide_index=True, width="stretch",
                         column_config={"similarity": st.column_config.ProgressColumn(
                             "similarity", min_value=0.0, max_value=1.0, format="%.3f")})
            st.caption("Cosine similarity of era-adjusted (per-season z-scored) stat profiles.")
        except KeyError as e:
            st.warning(f"Not enough data: {e}")

with tab_proj:
    if not model_available("player_lstm.pt"):
        st.info("Train the LSTM: `python -m ml.player_lstm train`")
    else:
        from ml.player_lstm import predict_next_season
        proj = predict_next_season(player.bref_id)
        if proj is None:
            st.info("Needs 4 recent seasons with 20+ games (at most one missed season).")
        else:
            st.markdown(f"Projection for **{season_label(proj['target_season'])}** "
                        f"(age {proj['age_next']}), from seasons "
                        f"{', '.join(map(season_label, proj['based_on_seasons']))}.")
            t = pd.DataFrame({"last season": proj["last_season"],
                              "projected": proj["predicted_next_season"]})
            t["change"] = t["projected"] - t["last season"]
            st.dataframe(t.style.format("{:+.3f}", subset=["change"]).format(
                "{:.3f}", subset=["last season", "projected"]), width="stretch")
            if proj["target_season"] in set(career.season):
                st.caption("This season has already been played; compare against the table.")

with tab_table:
    st.dataframe(career, hide_index=True, width="stretch")
