"""Shared Streamlit helpers: cached queries, Plotly chart helpers, dark theme."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from db.connection import query_df

ACCENT = "#2563eb"
PLOT_BG = "#0f0f0f"
TEXT = "#e5e7eb"
GRID = "#1f2937"
ARCHETYPE_COLORS = px.colors.qualitative.Bold

# Columns users may rank by; anything else is rejected (the v1 helper f-stringed `stat`).
LEADER_STATS = {
    "pts": "Points", "reb": "Rebounds", "ast": "Assists", "stl": "Steals", "blk": "Blocks",
    "fg3": "3PM", "fg_pct": "FG%", "fg3_pct": "3P%", "ft_pct": "FT%", "ts_pct": "TS%",
    "per": "PER", "bpm": "BPM", "ws": "Win Shares", "vorp": "VORP", "usg_pct": "USG%",
}
PCT_MIN_ATTEMPTS = {"fg_pct": ("fga", 5.0), "fg3_pct": ("fg3a", 2.0), "ft_pct": ("fta", 2.0)}


# ---------------------------------------------------------------- cached queries

@st.cache_data(ttl=3600)
def cached_player_list() -> pd.DataFrame:
    """bref_id, name and career span; the label disambiguates players sharing a name."""
    df = query_df("""
        SELECT p.bref_id, p.name, MIN(s.season) first, MAX(s.season) last,
               string_agg(DISTINCT s.team_list, ',') teams
        FROM players p JOIN player_seasons s ON s.player_id = p.id AND s.is_primary
        GROUP BY 1, 2 ORDER BY 2""")
    df["label"] = df["name"] + " (" + (df["first"] - 1).astype(str) + "–" + df["last"].astype(str) + ")"
    # Namesakes with the same span (e.g. two Tony Mitchells, both 2013-14) also need teams
    dup = df["label"].duplicated(keep=False)
    df.loc[dup, "label"] = df.loc[dup, "label"].str[:-1] + ", " + df.loc[dup, "teams"] + ")"
    dup = df["label"].duplicated(keep=False)
    df.loc[dup, "label"] = df.loc[dup, "label"] + " [" + df.loc[dup, "bref_id"] + "]"
    return df


@st.cache_data(ttl=3600)
def cached_team_list(season: int | None = None) -> pd.DataFrame:
    if season:
        return query_df("SELECT team_abr, full_name FROM team_seasons WHERE season = :s "
                        "ORDER BY full_name", {"s": season})
    return query_df("""SELECT franchise_id, current_abr AS team_abr, full_name
                       FROM franchises ORDER BY full_name""")


@st.cache_data(ttl=3600)
def cached_season_range() -> tuple[int, int]:
    r = query_df("SELECT MIN(season) lo, MAX(season) hi FROM player_seasons").iloc[0]
    return int(r.lo), int(r.hi)


@st.cache_data(ttl=3600)
def cached_player_career(bref_id: str) -> pd.DataFrame:
    return query_df("""
        SELECT s.season, s.team_abr, s.team_list, s.age, s.pos, s.g, s.gs, s.mp, s.pts, s.reb,
               s.ast, s.stl, s.blk, s.tov, s.fg_pct, s.fg3, s.fg3_pct, s.ft_pct, s.ts_pct,
               s.usg_pct, s.per, s.ws, s.bpm, s.vorp
        FROM player_seasons s JOIN players p ON p.id = s.player_id
        WHERE p.bref_id = :b AND s.is_primary ORDER BY s.season""", {"b": bref_id})


@st.cache_data(ttl=3600)
def cached_team_history(franchise_id: str) -> pd.DataFrame:
    df = query_df("""SELECT season, team_abr, full_name, conference, division, w, l, w_pct,
                            ps_g, pa_g, srs, made_playoffs
                     FROM team_seasons WHERE franchise_id = :f ORDER BY season""",
                  {"f": franchise_id})
    df["net_rating"] = df["ps_g"] - df["pa_g"]
    return df


@st.cache_data(ttl=3600)
def cached_leaders(season: int, stat: str = "pts", n: int = 15, min_games: int = 41) -> pd.DataFrame:
    if stat not in LEADER_STATS:
        raise ValueError(f"unsupported stat {stat!r}")
    extra = ""
    if stat in PCT_MIN_ATTEMPTS:
        col, minimum = PCT_MIN_ATTEMPTS[stat]
        extra = f"AND s.{col} >= {minimum}"
    return query_df(f"""
        SELECT p.name, s.team_abr, s.team_list, s.g, s.{stat} AS value
        FROM player_seasons s JOIN players p ON p.id = s.player_id
        WHERE s.season = :season AND s.is_primary AND s.g >= :g AND s.{stat} IS NOT NULL {extra}
        ORDER BY s.{stat} DESC LIMIT :n""", {"season": season, "g": min_games, "n": n})


@st.cache_data(ttl=3600)
def cached_league_averages() -> pd.DataFrame:
    """Minutes-weighted league averages per season (players with g ≥ 41)."""
    return query_df("""
        SELECT season,
               SUM(pts * g) / SUM(g)                AS pts,
               SUM(fg3a * g) / SUM(g)               AS fg3a,
               SUM(fg3 * g) / NULLIF(SUM(fg3a * g), 0) AS fg3_pct,
               SUM(ts_pct * mp_total) / SUM(mp_total) AS ts_pct,
               SUM(fg3a_rate * mp_total) / SUM(mp_total) AS fg3a_rate
        FROM player_seasons WHERE is_primary AND g >= 41 AND mp_total > 0
        GROUP BY season ORDER BY season""")


@st.cache_data(ttl=3600)
def cached_league_pace() -> pd.DataFrame:
    # home-win rate only where there is a home crowd: neutral sites would understate it
    return query_df("""SELECT season, AVG(home_score + away_score) AS total_pts,
                              AVG(home_win::int) FILTER (WHERE NOT neutral) AS home_win_rate,
                              COUNT(*) AS games
                       FROM games WHERE game_type = 'regular' GROUP BY 1 ORDER BY 1""")


@st.cache_data(ttl=3600)
def cached_counts() -> dict:
    r = query_df("""SELECT (SELECT COUNT(*) FROM players) players,
                           (SELECT COUNT(*) FROM games) games,
                           (SELECT COUNT(DISTINCT season) FROM games) seasons""")
    return r.iloc[0].to_dict()


# ---------------------------------------------------------------- charts

def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT), height=height, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def career_line(df: pd.DataFrame, y_col: str, title: str, y_label: str | None = None) -> go.Figure:
    fig = px.line(df, x="season", y=y_col, markers=True, title=title,
                  hover_data=[c for c in ("team_abr", "age", "g") if c in df.columns])
    fig.update_traces(line_color=ACCENT)
    fig.update_layout(yaxis_title=y_label or y_col, xaxis_title=None)
    return style_fig(fig, 300)


def stat_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    fig = px.bar(df, x=x_col, y=y_col, orientation="h", title=title,
                 hover_data=[c for c in ("team_abr", "g") if c in df.columns])
    fig.update_traces(marker_color=ACCENT)
    fig.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis_title=None)
    return style_fig(fig, max(320, 28 * len(df)))


def win_gauge(prob: float, label: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "valueformat": ".1f"},
        title={"text": label},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": ACCENT},
               "bgcolor": GRID, "threshold": {"line": {"color": TEXT, "width": 2}, "value": 50}},
    ))
    return style_fig(fig, 260)


# ---------------------------------------------------------------- theme

def apply_dark_theme():
    st.markdown("""
    <style>
      .stApp { background-color: #0a0a0a; }
      section[data-testid="stSidebar"] { background-color: #111111; }
      .metric-card { background: #111827; border: 1px solid #1f2937; border-radius: 10px;
                     padding: 14px 16px; margin-bottom: 8px; }
      .metric-card .label { color: #9ca3af; font-size: 0.8rem; text-transform: uppercase;
                            letter-spacing: .04em; }
      .metric-card .value { color: #e5e7eb; font-size: 1.6rem; font-weight: 600; }
      .badge { display: inline-block; padding: 4px 10px; border-radius: 999px;
               background: #1e3a8a; color: #dbeafe; font-weight: 600; }
    </style>""", unsafe_allow_html=True)


def metric_card(label: str, value) -> None:
    st.markdown(f'<div class="metric-card"><div class="label">{label}</div>'
                f'<div class="value">{value}</div></div>', unsafe_allow_html=True)


def model_available(path_name: str) -> bool:
    from pathlib import Path
    return (Path(__file__).resolve().parent.parent / "models" / path_name).exists()


def season_label(season: int) -> str:
    return f"{season - 1}-{str(season)[2:]}"


# ---------------------------------------------------------------- my predictions

def predicted_standings(picks: pd.DataFrame) -> pd.DataFrame:
    """Rank predicted wins within each conference, like the spreadsheet does.

    picks: team, conference ("East"/"West"), wins (blank = not picked yet), last_w.
    Ties are broken alphabetically. Seeds 1-6 make the playoffs, 7-10 the play-in.
    """
    df = picks.dropna(subset=["wins"]).sort_values(["conference", "wins", "team"],
                                                    ascending=[True, False, True])
    df["seed"] = df.groupby("conference").cumcount() + 1
    df["losses"] = 82 - df["wins"]
    df["gb"] = df.groupby("conference")["wins"].transform("max") - df["wins"]
    df["vs_last"] = df["wins"] - df["last_w"]
    df["status"] = pd.cut(df["seed"], [0, 6, 10, 15], labels=["Playoffs", "Play-in", "Lottery"])
    return df.reset_index(drop=True)
