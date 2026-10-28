"""Download Basketball-Reference pages, politely, with an on-disk cache.

Every page is saved to data/raw/{season}/{page}.html, so it is downloaded at most once;
re-parsing everything later needs no network. BRef allows about 20 requests/min and bans
clients that go faster for hours, so we sleep after every real request.
"""
import time
from pathlib import Path

import pandas as pd
import requests

from etl import parsers

BASE_URL = "https://www.basketball-reference.com/leagues"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; personal-research)"}
SLEEP_S = 4.0


class RateLimited(RuntimeError):
    pass


def fetch(url: str, cached: Path, refresh: bool = False, sleep_s: float = SLEEP_S) -> str | None:
    """HTML at `url`, read from / saved to `cached`. None if the server says 404.

    A 404 is remembered with an empty `.404` marker file next to the cache, so it isn't asked again.
    """
    missing = cached.with_suffix(".404")
    if not refresh:
        if cached.exists():
            return cached.read_text(encoding="utf-8")
        if missing.exists():
            return None

    response = requests.get(url, headers=HEADERS, timeout=30)
    time.sleep(sleep_s)
    if response.status_code == 429:
        raise RateLimited(f"429 from {url}: stop and wait before resuming")
    cached.parent.mkdir(parents=True, exist_ok=True)
    if response.status_code == 404:
        missing.touch()
        return None
    response.raise_for_status()
    response.encoding = "utf-8"
    cached.write_text(response.text, encoding="utf-8")
    return response.text


def fetch_page(season: int, page: str, refresh: bool = False, sleep_s: float = SLEEP_S) -> str | None:
    """A BRef league page: /leagues/NBA_{season}_{page}.html, cached as data/raw/{season}/{page}.html."""
    return fetch(f"{BASE_URL}/NBA_{season}_{page}.html", RAW_DIR / str(season) / f"{page}.html",
                 refresh, sleep_s)


def fetch_required(season: int, page: str, refresh: bool = False) -> str:
    html = fetch_page(season, page, refresh)
    if html is None:
        raise parsers.TableNotFound(f"page NBA_{season}_{page}.html does not exist")
    return html


def get_players(season: int, refresh: bool = False) -> pd.DataFrame:
    """Per-game and advanced stats merged into one wide row per (player, team)."""
    per_game = parsers.parse_player_table(fetch_required(season, "per_game", refresh), "per_game")
    advanced = parsers.parse_player_table(fetch_required(season, "advanced", refresh), "advanced")
    return parsers.merge_player_tables(per_game, advanced)


def get_standings(season: int, refresh: bool = False) -> pd.DataFrame:
    return parsers.parse_standings(fetch_required(season, "standings", refresh))


def get_schedule(season: int, refresh: bool = False) -> pd.DataFrame:
    """Every played game of the season, in date order.

    The season index NBA_{season}_games.html shows the first month and links to the other
    months, so the first month's own page is never fetched.
    """
    index = fetch_required(season, "games", refresh)
    months = parsers.parse_schedule_months(index, season)
    frames = [parsers.parse_schedule(index, season)]
    for slug in months[1:]:
        html = fetch_page(season, f"games-{slug}", refresh, sleep_s=5.0)
        if html is not None:
            frames.append(parsers.parse_schedule(html, season))
    games = pd.concat(frames, ignore_index=True)
    return games.sort_values(["game_date", "game_id"], ignore_index=True)


# ---------------------------------------------------------------- money: betting lines, payroll, cap

WIN_TOTALS_URL = "https://www.covers.com/sportsoddshistory/nba-win/?y={y1}-{y2}&sa=nba&t=win"
TEAM_URL = "https://www.basketball-reference.com/teams/{team}/{season}.html"
CAP_URL = "https://www.basketball-reference.com/contracts/salary-cap-history.html"


def get_win_totals(season: int, refresh: bool = False) -> pd.DataFrame:
    """Preseason win totals by team name (empty for seasons without lines, e.g. 1999)."""
    html = fetch(WIN_TOTALS_URL.format(y1=season - 1, y2=season),
                 RAW_DIR / "odds" / f"win_totals_{season}.html", refresh, sleep_s=3.0)  # site asks 3 s
    return parsers.parse_win_totals(html or "")


def get_salaries(season: int, refresh: bool = False) -> tuple[pd.DataFrame, int | None]:
    """(every team's payroll rows for the season, that season's salary cap).

    The season's teams come from its cached standings page; each team page is ~1 request.
    """
    teams = parsers.parse_standings(fetch_required(season, "standings"))["team_abr"]
    frames = []
    for team in teams:
        html = fetch(TEAM_URL.format(team=team, season=season),
                     RAW_DIR / str(season) / f"team_{team}.html", refresh)
        if html is None:
            raise parsers.TableNotFound(f"no team page for {team} {season}")
        frames.append(parsers.parse_salaries(html).assign(team_abr=team))
    caps = parsers.parse_salary_caps(fetch(CAP_URL, RAW_DIR / "salary_cap_history.html", refresh) or "")
    return pd.concat(frames, ignore_index=True), caps.get(season)


def get_roster(season: int, team: str, refresh: bool = False) -> list[str]:
    """Players on a team's roster page (for an unfinished season: today's roster)."""
    html = fetch(TEAM_URL.format(team=team, season=season),
                 RAW_DIR / str(season) / f"team_{team}.html", refresh)
    if html is None:
        raise parsers.TableNotFound(f"no team page for {team} {season}")
    return parsers.parse_roster(html)
