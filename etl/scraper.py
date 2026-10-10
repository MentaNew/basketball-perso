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


def fetch_page(season: int, page: str, refresh: bool = False, sleep_s: float = SLEEP_S) -> str | None:
    """HTML of /leagues/NBA_{season}_{page}.html, or None if BRef has no such page (404).

    A 404 is remembered with an empty `{page}.404` marker file, so it isn't asked again.
    """
    cached = RAW_DIR / str(season) / f"{page}.html"
    missing = cached.with_suffix(".404")
    if not refresh:
        if cached.exists():
            return cached.read_text(encoding="utf-8")
        if missing.exists():
            return None

    url = f"{BASE_URL}/NBA_{season}_{page}.html"
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
