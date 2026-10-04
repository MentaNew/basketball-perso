"""Pure HTML -> DataFrame parsers for Basketball-Reference pages (no network).

Cells are read by their `data-stat` attribute rather than header text: BRef renames
visible headers (e.g. "Tm" -> "Team") more often than it renames data-stat keys, and
reading by key makes visitor/home columns unambiguous (bug §7.1 in the blueprint).
"""
import re
import unicodedata
from datetime import date

import pandas as pd
from bs4 import BeautifulSoup, Comment, Tag

COMBINED_TEAM_RE = re.compile(r"^(TOT|\dTM)$")
TEAM_HREF_RE = re.compile(r"/teams/([A-Z0-9]{3})/")
MONTH_HREF_RE = re.compile(r"/leagues/NBA_(\d{4})_games-([a-z0-9-]+)\.html")


class TableNotFound(ValueError):
    pass


# ---------------------------------------------------------------- generic helpers

def find_table(html: str, table_id: str) -> Tag:
    """Find a table by id, including tables BRef hides inside HTML comments."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id=table_id)
    if table is not None:
        return table
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment) and table_id in s):
        table = BeautifulSoup(comment, "lxml").find("table", id=table_id)
        if table is not None:
            return table
    raise TableNotFound(f"table id={table_id!r} not found")


def table_ids(html: str) -> list[str]:
    return re.findall(r'<table[^>]*\bid="([^"]+)"', html)


def _cell_record(tr: Tag) -> dict:
    """Map data-stat -> text, plus '<stat>@href', '<stat>@csk' and '<stat>@id' extras."""
    rec = {}
    for cell in tr.find_all(["th", "td"]):
        stat = cell.get("data-stat")
        if not stat:
            continue
        rec[stat] = cell.get_text(strip=True)
        if cell.a and cell.a.get("href"):
            rec[f"{stat}@href"] = cell.a["href"]
        if cell.get("csk"):
            rec[f"{stat}@csk"] = cell["csk"]
        if cell.get("data-append-csv"):
            rec[f"{stat}@id"] = cell["data-append-csv"]
    return rec


def _body_rows(table: Tag) -> list[Tag]:
    body = table.tbody or table
    return [tr for tr in body.find_all("tr") if "thead" not in (tr.get("class") or [])]


def _first(rec: dict, *keys: str, default=None):
    for k in keys:
        if rec.get(k) not in (None, ""):
            return rec[k]
    return default


def _num(value) -> float | None:
    if value in (None, "", "—", "-"):
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def _team_from_href(href: str | None) -> str | None:
    m = TEAM_HREF_RE.search(href or "")
    return m.group(1) if m else None


def clean_name(name: str) -> str:
    return name.replace("*", "").strip()


def search_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", ascii_name).strip().lower()


# ---------------------------------------------------------------- player tables

# data-stat key(s) -> our column.  Alternatives cover older BRef layouts.
PER_GAME_MAP = {
    "g": ("games", "g"), "gs": ("games_started", "gs"), "mp": ("mp_per_g",),
    "fg": ("fg_per_g",), "fga": ("fga_per_g",), "fg_pct": ("fg_pct",),
    "fg3": ("fg3_per_g",), "fg3a": ("fg3a_per_g",), "fg3_pct": ("fg3_pct",),
    "ft": ("ft_per_g",), "fta": ("fta_per_g",), "ft_pct": ("ft_pct",), "efg_pct": ("efg_pct",),
    "orb": ("orb_per_g",), "drb": ("drb_per_g",), "reb": ("trb_per_g",),
    "ast": ("ast_per_g",), "stl": ("stl_per_g",), "blk": ("blk_per_g",),
    "tov": ("tov_per_g",), "pf": ("pf_per_g",), "pts": ("pts_per_g",),
}
ADVANCED_MAP = {
    "mp_total": ("mp",), "per": ("per",), "ts_pct": ("ts_pct",),
    "fg3a_rate": ("fg3a_per_fga_pct",), "fta_rate": ("fta_per_fga_pct",),
    "orb_pct": ("orb_pct",), "drb_pct": ("drb_pct",), "trb_pct": ("trb_pct",),
    "ast_pct": ("ast_pct",), "stl_pct": ("stl_pct",), "blk_pct": ("blk_pct",),
    "tov_pct": ("tov_pct",), "usg_pct": ("usg_pct",), "ows": ("ows",), "dws": ("dws",),
    "ws": ("ws",), "ws_per_48": ("ws_per_48",), "obpm": ("obpm",), "dbpm": ("dbpm",),
    "bpm": ("bpm",), "vorp": ("vorp",),
}
PLAYER_TABLE_IDS = {
    "per_game": ("per_game_stats",),
    "advanced": ("advanced", "advanced_stats"),
}


def parse_player_table(html: str, kind: str) -> pd.DataFrame:
    """Parse a league per_game or advanced page into one row per (player, team).

    Columns: bref_id, name, age, team_abr ('TOT' for combined rows), pos, team_list,
    is_primary, plus the stat columns of `kind`.
    """
    stat_map = PER_GAME_MAP if kind == "per_game" else ADVANCED_MAP
    table = None
    for tid in PLAYER_TABLE_IDS[kind]:
        try:
            table = find_table(html, tid)
            break
        except TableNotFound:
            continue
    if table is None:
        raise TableNotFound(f"no {kind} table (tried {PLAYER_TABLE_IDS[kind]})")

    rows = []
    for tr in _body_rows(table):
        rec = _cell_record(tr)
        bref_id = _first(rec, "name_display@id", "player@id")
        if not bref_id:
            continue  # league-average / separator rows
        team = _first(rec, "team_name_abbr", "team_id", default="UNK")
        row = {
            "bref_id": bref_id,
            "name": clean_name(_first(rec, "name_display", "player", default="")),
            "age": _num(rec.get("age")),
            "team_abr": "TOT" if COMBINED_TEAM_RE.match(team) else team,
            "pos": rec.get("pos") or None,
        }
        for col, keys in stat_map.items():
            row[col] = _num(_first(rec, *keys))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # One primary row per player-season: the TOT row for traded players, else the only row.
    has_tot = df.groupby("bref_id")["team_abr"].transform(lambda s: (s == "TOT").any())
    df["is_primary"] = (df["team_abr"] == "TOT") | ~has_tot
    teams = (df[df["team_abr"] != "TOT"].groupby("bref_id", sort=False)["team_abr"]
             .agg(",".join))
    df["team_list"] = df["bref_id"].map(teams).where(df["team_abr"] == "TOT", df["team_abr"])
    return df.drop_duplicates(["bref_id", "team_abr"]).reset_index(drop=True)


# ---------------------------------------------------------------- standings

def parse_standings(html: str) -> pd.DataFrame:
    """Parse both conferences.  Uses division tables (present in every era) and falls back
    to conference tables; each table is normalised on its own before concatenation (§7.3)."""
    frames = []
    for conf in ("E", "W"):
        table = tid = None
        for cand in (f"divs_standings_{conf}", f"confs_standings_{conf}"):
            try:
                table, tid = find_table(html, cand), cand
                break
            except TableNotFound:
                continue
        if table is None:
            raise TableNotFound(f"no standings table for conference {conf}")
        division = None
        rows = []
        body = table.tbody or table
        for tr in body.find_all("tr"):
            if "thead" in (tr.get("class") or []):
                division = tr.get_text(strip=True).replace(" Division", "") or division
                continue
            rec = _cell_record(tr)
            team_abr = _team_from_href(rec.get("team_name@href"))
            if not team_abr:
                continue
            rows.append({
                "team_abr": team_abr,
                "full_name": clean_name(rec.get("team_name", "")),
                "made_playoffs": "*" in rec.get("team_name", ""),
                "conference": conf,
                "division": division if tid.startswith("divs") else None,
                "w": _num(rec.get("wins")), "l": _num(rec.get("losses")),
                "w_pct": _num(rec.get("win_loss_pct")),
                "gb": _num(rec.get("gb")) or 0.0,
                "ps_g": _num(rec.get("pts_per_g")), "pa_g": _num(rec.get("opp_pts_per_g")),
                "srs": _num(rec.get("srs")),
            })
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)
    for c in ("w", "l"):
        df[c] = df[c].astype("Int64")
    return df


# ---------------------------------------------------------------- schedule

def parse_schedule_months(html: str, season: int) -> list[str]:
    """Month slugs linked from NBA_{season}_games.html (e.g. 'october-2019', 'july')."""
    slugs = []
    for y, slug in MONTH_HREF_RE.findall(html):
        if int(y) == season and slug not in slugs:
            slugs.append(slug)
    return slugs


def parse_schedule(html: str, season: int) -> pd.DataFrame:
    """One row per *played* game.  BRef columns are visitor first, then home (§7.1)."""
    table = find_table(html, "schedule")
    rows = []
    for tr in _body_rows(table):
        rec = _cell_record(tr)
        away_score, home_score = _num(rec.get("visitor_pts")), _num(rec.get("home_pts"))
        if away_score is None or home_score is None:
            continue  # unplayed / postponed
        away = _team_from_href(rec.get("visitor_team_name@href"))
        home = _team_from_href(rec.get("home_team_name@href"))
        game_id = rec.get("date_game@csk") or ""
        box = rec.get("box_score_text@href", "")
        if not re.fullmatch(r"\d{9}[A-Z]{3}", game_id):
            m = re.search(r"/boxscores/(\w+)\.html", box)
            game_id = m.group(1) if m else f"{season}_{rec.get('date_game')}_{away}_{home}"
        game_date = _parse_date(game_id, rec.get("date_game", ""))
        rows.append({
            "game_id": game_id,
            "season": season,
            "game_date": game_date,
            "away_team": away, "home_team": home,
            "away_name": rec.get("visitor_team_name"), "home_name": rec.get("home_team_name"),
            "away_score": int(away_score), "home_score": int(home_score),
            "home_win": home_score > away_score,
            "overtimes": rec.get("overtimes") or None,
            "attendance": int(_num(rec.get("attendance"))) if _num(rec.get("attendance")) else None,
            "remarks": rec.get("game_remarks") or None,
            "neutral": is_neutral_site(rec.get("game_remarks"), season, game_date),
        })
    return pd.DataFrame(rows)


BUBBLE_START = date(2020, 7, 30)        # 2020 restart in Orlando: no home crowds
VENUE_RE = re.compile(r"^at [^,]+, ([^(]+)")


def is_neutral_site(remarks: str | None, season: int, game_date: date) -> bool:
    """True for games abroad (London, Mexico City, Tokyo, Paris, ...) and the 2020 bubble.

    Domestic alternate venues ("at Hartford, CT" for the Celtics) and Canadian ones (Raptors
    and Grizzlies home alternates) keep a real home edge in the data, so they stay home games.
    """
    if season == 2020 and game_date >= BUBBLE_START:
        return True
    m = VENUE_RE.match(remarks or "")
    if not m:
        return False
    place = m.group(1).strip()
    return not re.fullmatch(r"[A-Z]{2}", place) and place != "Canada"


def _parse_date(game_id: str, text: str) -> date:
    if re.match(r"\d{8}", game_id):
        return date(int(game_id[:4]), int(game_id[4:6]), int(game_id[6:8]))
    return pd.to_datetime(text, format="%a, %b %d, %Y").date()
