"""Turn Basketball-Reference HTML into DataFrames. Pure functions: no network, no database.

Every BRef table cell has a stable `data-stat` attribute, e.g.

    <td data-stat="visitor_pts">107</td> ... <td data-stat="home_pts">119</td>

We read cells by that key, never by column position or header text. Headers get renamed
("Tm" became "Team"), and position-based reading is how v1 swapped home and away scores.
"""
import re
import unicodedata
from datetime import date

import pandas as pd
from bs4 import BeautifulSoup, Tag


class TableNotFound(ValueError):
    pass


# ---------------------------------------------------------------- helpers

def find_table(html: str, table_id: str) -> Tag:
    table = BeautifulSoup(html, "lxml").find("table", id=table_id)
    if table is None:
        raise TableNotFound(f"table id={table_id!r} not found")
    return table


def read_row(tr: Tag) -> dict[str, str]:
    """{data-stat: text} for one table row, plus three extras we need:
    '<stat>@href' (link target), '<stat>@csk' (sort key) and '<stat>@id' (player id)."""
    row = {}
    for cell in tr.find_all(["th", "td"]):
        stat = cell.get("data-stat")
        if not stat:
            continue
        row[stat] = cell.get_text(strip=True)
        if cell.a:
            row[f"{stat}@href"] = cell.a["href"]
        if cell.has_attr("csk"):
            row[f"{stat}@csk"] = cell["csk"]
        if cell.has_attr("data-append-csv"):
            row[f"{stat}@id"] = cell["data-append-csv"]
    return row


def to_float(text: str | None) -> float | None:
    """'1,234' -> 1234.0; '' or '—' -> None."""
    try:
        return float(text.replace(",", ""))
    except (AttributeError, ValueError):
        return None


def team_from_link(href: str | None) -> str | None:
    """'/teams/OKC/2025.html' -> 'OKC'."""
    match = re.search(r"/teams/([A-Z0-9]{3})/", href or "")
    return match.group(1) if match else None


def clean_name(name: str) -> str:
    """Drop the Hall-of-Fame marker: 'Michael Jordan*' -> 'Michael Jordan'."""
    return name.replace("*", "").strip()


def search_name(name: str) -> str:
    """Lower-case, accent-free form for lookups: 'Nikola Jokić' -> 'nikola jokic'."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(ascii_name.split()).lower()


# ---------------------------------------------------------------- player tables

# our column -> BRef data-stat, per page
PER_GAME_STATS = {
    "g": "games", "gs": "games_started", "mp": "mp_per_g",
    "fg": "fg_per_g", "fga": "fga_per_g", "fg_pct": "fg_pct",
    "fg3": "fg3_per_g", "fg3a": "fg3a_per_g", "fg3_pct": "fg3_pct",
    "ft": "ft_per_g", "fta": "fta_per_g", "ft_pct": "ft_pct", "efg_pct": "efg_pct",
    "orb": "orb_per_g", "drb": "drb_per_g", "reb": "trb_per_g",
    "ast": "ast_per_g", "stl": "stl_per_g", "blk": "blk_per_g",
    "tov": "tov_per_g", "pf": "pf_per_g", "pts": "pts_per_g",
}
ADVANCED_STATS = {
    "mp_total": "mp", "per": "per", "ts_pct": "ts_pct",
    "fg3a_rate": "fg3a_per_fga_pct", "fta_rate": "fta_per_fga_pct",
    "orb_pct": "orb_pct", "drb_pct": "drb_pct", "trb_pct": "trb_pct",
    "ast_pct": "ast_pct", "stl_pct": "stl_pct", "blk_pct": "blk_pct",
    "tov_pct": "tov_pct", "usg_pct": "usg_pct", "ows": "ows", "dws": "dws",
    "ws": "ws", "ws_per_48": "ws_per_48", "obpm": "obpm", "dbpm": "dbpm",
    "bpm": "bpm", "vorp": "vorp",
}
PLAYER_PAGES = {  # kind -> (table id, stat map)
    "per_game": ("per_game_stats", PER_GAME_STATS),
    "advanced": ("advanced", ADVANCED_STATS),
}


def parse_player_table(html: str, kind: str) -> pd.DataFrame:
    """One row per (player, team) from a league per_game or advanced page.

    A traded player appears once per team plus once combined. BRef labels the combined
    row 'TOT' (older pages) or '2TM'/'3TM' (current pages); we always call it 'TOT'.
    Adds `is_primary` (the TOT row if any, otherwise the only row) and `team_list`.
    """
    table_id, stats = PLAYER_PAGES[kind]
    rows = []
    for tr in find_table(html, table_id).tbody.find_all("tr"):
        cell = read_row(tr)
        if "name_display@id" not in cell:
            continue  # the "League Average" row has no player id
        team = cell["team_name_abbr"]
        rows.append({
            "bref_id": cell["name_display@id"],
            "name": clean_name(cell["name_display"]),
            "age": to_float(cell.get("age")),
            "team_abr": "TOT" if re.fullmatch(r"TOT|\dTM", team) else team,
            "pos": cell.get("pos") or None,
            **{col: to_float(cell.get(key)) for col, key in stats.items()},
        })
    df = pd.DataFrame(rows)

    traded = set(df.loc[df["team_abr"] == "TOT", "bref_id"])
    df["is_primary"] = (df["team_abr"] == "TOT") | ~df["bref_id"].isin(traded)
    teams_of = df[df["team_abr"] != "TOT"].groupby("bref_id", sort=False)["team_abr"].agg(",".join)
    df["team_list"] = df["team_abr"].where(df["team_abr"] != "TOT", df["bref_id"].map(teams_of))
    return df


def merge_player_tables(per_game: pd.DataFrame, advanced: pd.DataFrame) -> pd.DataFrame:
    """Join the two player pages into one wide row per (player, team).

    Both pages list exactly the same rows (checked on every season since 1980). The outer
    join with validate="one_to_one" still guarantees nothing is silently dropped or doubled.
    """
    advanced_stats = advanced[["bref_id", "team_abr", *ADVANCED_STATS]]
    return per_game.merge(advanced_stats, on=["bref_id", "team_abr"], how="outer",
                          validate="one_to_one")


# ---------------------------------------------------------------- standings

def parse_standings(html: str) -> pd.DataFrame:
    """Both conferences' final standings, one row per team.

    The division tables (divs_standings_E/W) exist in every season since 1980. Each
    conference is parsed on its own; v1 merged the two tables first and lost the West.
    Inside a table, rows with class "thead" are division headers ("Atlantic Division").
    """
    rows = []
    for conference in ("E", "W"):
        division = None
        for tr in find_table(html, f"divs_standings_{conference}").tbody.find_all("tr"):
            if "thead" in tr.get("class", []):
                division = tr.get_text(strip=True).removesuffix(" Division")
                continue
            cell = read_row(tr)
            rows.append({
                "team_abr": team_from_link(cell["team_name@href"]),
                "full_name": clean_name(cell["team_name"]),
                "made_playoffs": cell["team_name"].endswith("*"),
                "conference": conference,
                "division": division,
                "w": int(cell["wins"]),
                "l": int(cell["losses"]),
                "w_pct": to_float(cell["win_loss_pct"]),
                "gb": to_float(cell["gb"]) or 0.0,  # the leader shows '—'
                "ps_g": to_float(cell["pts_per_g"]),
                "pa_g": to_float(cell["opp_pts_per_g"]),
                "srs": to_float(cell["srs"]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- schedule

def parse_schedule_months(html: str, season: int) -> list[str]:
    """Month page slugs linked from NBA_{season}_games.html, in order.

    Discovered rather than hard-coded because seasons vary: lockouts skip months, and the
    2020 bubble ran 'october-2019' ... 'july', 'august', 'september', 'october-2020'.
    """
    slugs = re.findall(rf"/leagues/NBA_{season}_games-([a-z0-9-]+)\.html", html)
    return list(dict.fromkeys(slugs))  # de-duplicate, keep order


def parse_schedule(html: str, season: int) -> pd.DataFrame:
    """One row per played game on a schedule page (columns: visitor first, then home)."""
    rows = []
    for tr in find_table(html, "schedule").tbody.find_all("tr"):
        cell = read_row(tr)
        if not cell.get("home_pts"):
            continue  # scheduled but not played yet
        game_id = cell["date_game@csk"]  # e.g. "202310240DEN": date + "0" + home team
        game_date = date(int(game_id[:4]), int(game_id[4:6]), int(game_id[6:8]))
        away_score, home_score = int(cell["visitor_pts"]), int(cell["home_pts"])
        attendance = to_float(cell.get("attendance"))
        rows.append({
            "game_id": game_id,
            "season": season,
            "game_date": game_date,
            "away_team": team_from_link(cell["visitor_team_name@href"]),
            "home_team": team_from_link(cell["home_team_name@href"]),
            "away_name": cell["visitor_team_name"],
            "home_name": cell["home_team_name"],
            "away_score": away_score,
            "home_score": home_score,
            "home_win": home_score > away_score,
            "overtimes": cell.get("overtimes") or None,
            "attendance": int(attendance) if attendance else None,
            "remarks": cell.get("game_remarks") or None,
            "neutral": is_neutral_site(cell.get("game_remarks"), season, game_date),
        })
    return pd.DataFrame(rows)


BUBBLE_START = date(2020, 7, 30)  # the 2020 restart in Orlando: no home crowds


def is_neutral_site(remarks: str | None, season: int, game_date: date) -> bool:
    """True for games abroad ("at London, England", "at Mexico City, Mexico") and the bubble.

    Domestic alternate venues ("at Hartford, CT", a Celtics second home) and Canadian ones
    (Raptors/Grizzlies alternates) keep a real home edge in the data, so they stay home games.
    """
    if season == 2020 and game_date >= BUBBLE_START:
        return True
    match = re.match(r"at [^,]+, ([^(]+)", remarks or "")
    if not match:
        return False
    place = match.group(1).strip()          # "England", "CT", "Canada"
    is_us_state = re.fullmatch(r"[A-Z]{2}", place) is not None
    return not is_us_state and place != "Canada"
