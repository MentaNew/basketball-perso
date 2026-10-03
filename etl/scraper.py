"""Polite, cached fetching of Basketball-Reference pages.

Every page is cached to data/raw/{season}/{page}.html so parsing can be re-run without
re-scraping.  BRef allows ~20 requests/min and bans offenders for hours, so we sleep
after every network request.
"""
import time
from pathlib import Path

import pandas as pd
import requests

from etl import parsers

BASE_URL = "https://www.basketball-reference.com/leagues"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; personal-research)",
    "Accept-Language": "en-US,en;q=0.9",
}
SLEEP_S = 4.0


class RateLimited(RuntimeError):
    pass


def fetch_page(season: int, page: str, refresh: bool = False, sleep_s: float = SLEEP_S) -> str | None:
    """Return the HTML of /leagues/NBA_{season}_{page}.html, or None on 404."""
    cache = RAW_DIR / str(season) / f"{page}.html"
    missing_marker = cache.with_suffix(".404")
    if not refresh:
        if cache.exists():
            return cache.read_text(encoding="utf-8")
        if missing_marker.exists():
            return None

    url = f"{BASE_URL}/NBA_{season}_{page}.html"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    time.sleep(sleep_s)
    if resp.status_code == 429:
        raise RateLimited(f"429 from {url}: stop and wait before resuming")
    cache.parent.mkdir(parents=True, exist_ok=True)
    if resp.status_code == 404:
        missing_marker.touch()
        return None
    resp.raise_for_status()
    resp.encoding = "utf-8"
    cache.write_text(resp.text, encoding="utf-8")
    return resp.text


def _require(html: str | None, season: int, page: str) -> str:
    if html is None:
        raise parsers.TableNotFound(f"page NBA_{season}_{page}.html does not exist")
    return html


def get_player_per_game(season: int, refresh: bool = False) -> pd.DataFrame:
    html = _require(fetch_page(season, "per_game", refresh), season, "per_game")
    return parsers.parse_player_table(html, "per_game")


def get_player_advanced(season: int, refresh: bool = False) -> pd.DataFrame:
    html = _require(fetch_page(season, "advanced", refresh), season, "advanced")
    return parsers.parse_player_table(html, "advanced")


def get_standings(season: int, refresh: bool = False) -> pd.DataFrame:
    html = _require(fetch_page(season, "standings", refresh), season, "standings")
    return parsers.parse_standings(html)


def get_season_schedule(season: int, refresh: bool = False) -> pd.DataFrame:
    """All played games of a season.  Month pages are discovered from the season index,
    which handles lockouts and the 2020 bubble (october-2019 ... october-2020)."""
    index = _require(fetch_page(season, "games", refresh), season, "games")
    months = parsers.parse_schedule_months(index, season)
    frames = [parsers.parse_schedule(index, season)]  # index page shows the first month
    for slug in months:
        html = fetch_page(season, f"games-{slug}", refresh, sleep_s=5.0)
        if html is not None:
            frames.append(parsers.parse_schedule(html, season))
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return df.drop_duplicates("game_id").sort_values(["game_date", "game_id"], ignore_index=True)
