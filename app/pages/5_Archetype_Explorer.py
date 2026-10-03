import plotly.express as px
import streamlit as st

from app.utils import (
    ARCHETYPE_COLORS,
    apply_dark_theme,
    model_available,
    season_label,
    style_fig,
)

st.set_page_config(page_title="Archetype Explorer", page_icon="🏀", layout="wide")
apply_dark_theme()
st.title("Archetype Explorer")

if not model_available("similarity.pkl"):
    st.info("Train the similarity model first: `python -m ml.similarity train`")
    st.stop()

from ml.similarity import get_archetypes, get_umap_plot_data  # noqa: E402

data = get_umap_plot_data()
lo, hi = int(data.season.min()), int(data.season.max())
c1, c2 = st.columns([3, 2])
rng = c1.slider("Seasons", lo, hi, (lo, hi), format="%d")
search = c2.text_input("Highlight player", placeholder="e.g. Stephen Curry")
view = data[data.season.between(*rng)]

order = sorted(data.archetype.unique())
fig = px.scatter(
    view, x="umap_x", y="umap_y", color="archetype", category_orders={"archetype": order},
    color_discrete_sequence=ARCHETYPE_COLORS, opacity=0.55,
    hover_name="name",
    hover_data={"season": True, "team_abr": True, "pts": ":.1f", "reb": ":.1f", "ast": ":.1f",
                "ts_pct": ":.3f", "umap_x": False, "umap_y": False},
)
fig.update_traces(marker=dict(size=5))
if search:
    hit = view[view.name.str.contains(search, case=False, na=False)]
    if not hit.empty:
        fig.add_scatter(x=hit.umap_x, y=hit.umap_y, mode="markers+text",
                        text=hit.season.map(season_label), textposition="top center",
                        marker=dict(size=12, color="white", line=dict(color="black", width=1)),
                        name=hit.name.iat[0], hoverinfo="skip")
fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                  title=f"{len(view):,} player-seasons (41+ games), era-adjusted")
st.plotly_chart(style_fig(fig, 650), width="stretch")

c1, c2 = st.columns(2)
with c1:
    share = (view.groupby(["season", "archetype"]).size()
             .groupby(level=0).transform(lambda s: s / s.sum()).rename("share").reset_index())
    fig = px.area(share, x="season", y="share", color="archetype",
                  category_orders={"archetype": order}, color_discrete_sequence=ARCHETYPE_COLORS,
                  title="Archetype mix by season")
    fig.update_layout(yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
    st.plotly_chart(style_fig(fig, 400), width="stretch")
with c2:
    st.markdown("**Cluster centroids** (mean per-season z-score)")
    arch = get_archetypes().drop(columns="cluster").set_index("archetype")
    st.dataframe(arch.style.format("{:+.2f}", subset=arch.columns[1:]), width="stretch")
st.caption("Names are assigned by matching each cluster centroid to archetype templates, "
           "so they follow the data rather than arbitrary cluster ids.")
