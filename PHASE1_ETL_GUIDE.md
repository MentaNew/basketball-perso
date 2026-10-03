# Phase 1 — ETL Pipeline & Database Setup

Build the data foundation: scrape 45 seasons of NBA data from Basketball-Reference,
store it in PostgreSQL, and have a clean, queryable database ready for the ML models.

---

## Step 0 — Install PostgreSQL

PostgreSQL is a database server that runs as a background process on your Mac.
Install it once via Homebrew, then it runs in the background like any service.

```bash
brew install postgresql@17
brew services start postgresql@17

# Verify it's running
psql postgres -c "SELECT version();"
```

Create the project database and user:

```sql
-- Run this inside psql
CREATE DATABASE nba_platform;
CREATE USER nba_user WITH PASSWORD 'yourpassword';
GRANT ALL PRIVILEGES ON DATABASE nba_platform TO nba_user;
```

Your connection string (goes in `.env`):
```
DATABASE_URL=postgresql://nba_user:yourpassword@localhost:5432/nba_platform
```

---

## Step 1 — Python Environment

```bash
cd BasketBall
conda create -n nba python=3.11 -y
conda activate nba

pip install requests beautifulsoup4 pandas sqlalchemy psycopg2-binary \
            python-dotenv tqdm pyyaml
```

---

## Step 2 — Project Structure to Build

```
BasketBall/
├── .env                  ← DATABASE_URL (never commit)
├── .env.example
├── requirements.txt
│
├── db/
│   ├── schema.sql        ← Run once to create all tables
│   └── models.py         ← SQLAlchemy ORM models (mirrors schema.sql)
│
└── etl/
    ├── scraper.py        ← Extended from existing scrapping.py
    ├── pipeline.py       ← Orchestrates the full 45-season scrape
    └── loader.py         ← Bulk inserts scraped DataFrames into PostgreSQL
```

---

## Step 3 — Database Schema (`db/schema.sql`)

Run this once against your database:
```bash
psql $DATABASE_URL -f db/schema.sql
```

```sql
-- db/schema.sql

CREATE TABLE IF NOT EXISTS players (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    bref_id     TEXT UNIQUE,          -- Basketball-Reference player ID
    position    TEXT,
    birth_year  INT
);

CREATE TABLE IF NOT EXISTS teams (
    id           SERIAL PRIMARY KEY,
    abbreviation TEXT UNIQUE NOT NULL, -- "LAL", "BOS", etc.
    name         TEXT
);

CREATE TABLE IF NOT EXISTS player_season_stats (
    id          SERIAL PRIMARY KEY,
    player_id   INT REFERENCES players(id),
    team_abr    TEXT,
    season      INT NOT NULL,          -- ending year: 2024 = 2023-24 season
    stat_type   TEXT NOT NULL,         -- 'per_game', 'totals', 'advanced'

    -- Counting stats
    g           FLOAT,  -- games played
    gs          FLOAT,  -- games started
    mp          FLOAT,  -- minutes per game (or total)
    pts         FLOAT,
    reb         FLOAT,
    ast         FLOAT,
    stl         FLOAT,
    blk         FLOAT,
    tov         FLOAT,
    pf          FLOAT,

    -- Shooting
    fg          FLOAT,  fg_pct FLOAT,
    fg3         FLOAT,  fg3_pct FLOAT,
    ft          FLOAT,  ft_pct FLOAT,

    -- Advanced (per_game rows will have these as NULL)
    per         FLOAT,  -- Player Efficiency Rating
    ts_pct      FLOAT,  -- True Shooting %
    usg_pct     FLOAT,  -- Usage %
    ws          FLOAT,  -- Win Shares
    bpm         FLOAT,  -- Box Plus/Minus
    vorp        FLOAT,  -- Value Over Replacement

    UNIQUE (player_id, season, team_abr, stat_type)
);

CREATE TABLE IF NOT EXISTS team_season_stats (
    id           SERIAL PRIMARY KEY,
    team_abr     TEXT NOT NULL,
    season       INT NOT NULL,
    conference   TEXT,  -- 'E' or 'W'
    w            INT,   -- wins
    l            INT,   -- losses
    w_pct        FLOAT,
    gb           FLOAT, -- games behind
    ps_g         FLOAT, -- points scored per game
    pa_g         FLOAT, -- points allowed per game
    UNIQUE (team_abr, season)
);

CREATE TABLE IF NOT EXISTS games (
    id           SERIAL PRIMARY KEY,
    season       INT NOT NULL,
    game_date    DATE NOT NULL,
    home_team    TEXT NOT NULL,
    away_team    TEXT NOT NULL,
    home_score   INT,
    away_score   INT,
    home_win     BOOLEAN,
    UNIQUE (season, game_date, home_team, away_team)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_pss_player   ON player_season_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_pss_season   ON player_season_stats(season);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);
CREATE INDEX IF NOT EXISTS idx_games_date   ON games(game_date);
```

---

## Step 4 — SQLAlchemy Models (`db/models.py`)

This mirrors `schema.sql` as Python classes. Use this for querying from the ML code.

```python
# db/models.py
from sqlalchemy import (
    Column, Integer, Float, Text, Boolean, Date, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, relationship

class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "players"
    id         = Column(Integer, primary_key=True)
    name       = Column(Text, nullable=False)
    bref_id    = Column(Text, unique=True)
    position   = Column(Text)
    birth_year = Column(Integer)
    stats      = relationship("PlayerSeasonStats", back_populates="player")


class PlayerSeasonStats(Base):
    __tablename__ = "player_season_stats"
    __table_args__ = (
        UniqueConstraint("player_id", "season", "team_abr", "stat_type"),
    )
    id        = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("players.id"))
    team_abr  = Column(Text)
    season    = Column(Integer, nullable=False)
    stat_type = Column(Text, nullable=False)

    g = Column(Float); gs = Column(Float); mp = Column(Float)
    pts = Column(Float); reb = Column(Float); ast = Column(Float)
    stl = Column(Float); blk = Column(Float); tov = Column(Float)
    pf = Column(Float)
    fg = Column(Float); fg_pct = Column(Float)
    fg3 = Column(Float); fg3_pct = Column(Float)
    ft = Column(Float); ft_pct = Column(Float)
    per = Column(Float); ts_pct = Column(Float); usg_pct = Column(Float)
    ws = Column(Float); bpm = Column(Float); vorp = Column(Float)

    player = relationship("Player", back_populates="stats")


class TeamSeasonStats(Base):
    __tablename__ = "team_season_stats"
    __table_args__ = (UniqueConstraint("team_abr", "season"),)
    id         = Column(Integer, primary_key=True)
    team_abr   = Column(Text, nullable=False)
    season     = Column(Integer, nullable=False)
    conference = Column(Text)
    w = Column(Integer); l = Column(Integer); w_pct = Column(Float)
    gb = Column(Float); ps_g = Column(Float); pa_g = Column(Float)


class Game(Base):
    __tablename__ = "games"
    __table_args__ = (
        UniqueConstraint("season", "game_date", "home_team", "away_team"),
    )
    id         = Column(Integer, primary_key=True)
    season     = Column(Integer, nullable=False)
    game_date  = Column(Date, nullable=False)
    home_team  = Column(Text, nullable=False)
    away_team  = Column(Text, nullable=False)
    home_score = Column(Integer)
    away_score = Column(Integer)
    home_win   = Column(Boolean)
```

---

## Step 5 — Scraper Extension (`etl/scraper.py`)

Extend the existing `scrapping.py` with the additional table types you need.
The existing `fetch_league_table` function handles everything — just add these wrappers:

```python
# etl/scraper.py
# Import the existing helpers from scrapping.py (move scrapping.py into etl/ 
# or just copy the core functions here)

import time, requests, pandas as pd
from bs4 import BeautifulSoup

# --- Re-use your existing fetch_league_table + clean_basketball_reference_data ---
# (copy them here or: from scrapping import fetch_league_table)


def get_player_advanced(year: int) -> pd.DataFrame:
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    return fetch_league_table(url, table_id="advanced_stats")


def get_team_standings_both(year: int) -> pd.DataFrame:
    """Return both conferences combined."""
    east = get_team_standings(year, "E")
    east["conference"] = "E"
    west = get_team_standings(year, "W")
    west["conference"] = "W"
    return pd.concat([east, west], ignore_index=True)


def get_season_schedule(year: int) -> pd.DataFrame:
    """
    Scrape the full season schedule/results for a given year.
    Returns one row per game with home team, away team, scores.

    Basketball-Reference schedule pages are split by month.
    Months in an NBA season: October through June.
    """
    months = ["october", "november", "december", "january",
              "february", "march", "april", "may", "june"]

    # Seasons before 2002 started in November
    if year <= 2001:
        months = [m for m in months if m != "october"]

    all_games = []
    for month in months:
        url = (f"https://www.basketball-reference.com/leagues/"
               f"NBA_{year}_games-{month}.html")
        try:
            df = fetch_league_table(url, table_id="schedule", sleep_s=5.0)
            all_games.append(df)
        except ValueError:
            # Month doesn't exist for this season — normal, skip it
            pass
        except Exception as e:
            print(f"  Warning: {month} {year} failed — {e}")

    if not all_games:
        return pd.DataFrame()

    schedule = pd.concat(all_games, ignore_index=True)

    # Standardise column names (Basketball-Reference changes these occasionally)
    rename_map = {
        "Date": "game_date",
        "Home/Neutral": "home_team",
        "Visitor/Neutral": "away_team",
        "PTS": "home_score",
        "PTS.1": "away_score",   # away score is the second PTS column
    }
    schedule = schedule.rename(columns={
        k: v for k, v in rename_map.items() if k in schedule.columns
    })

    # Drop rows without scores (future games)
    schedule = schedule.dropna(subset=["home_score", "away_score"])
    schedule["home_score"] = pd.to_numeric(schedule["home_score"], errors="coerce")
    schedule["away_score"] = pd.to_numeric(schedule["away_score"], errors="coerce")
    schedule["home_win"] = schedule["home_score"] > schedule["away_score"]
    schedule["season"] = year

    return schedule[["season", "game_date", "home_team", "away_team",
                     "home_score", "away_score", "home_win"]]
```

---

## Step 6 — Loader (`etl/loader.py`)

```python
# etl/loader.py
"""
Bulk-insert scraped DataFrames into PostgreSQL.
All inserts use INSERT ... ON CONFLICT DO NOTHING (idempotent — safe to re-run).
"""
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    return create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)


def upsert_player_stats(df: pd.DataFrame, season: int, stat_type: str):
    """
    Insert player per-game, totals, or advanced stats for one season.
    stat_type: 'per_game' | 'totals' | 'advanced'
    """
    engine = get_engine()
    COL_MAP = {
        "Player": "name",  "Pos": "position", "Tm": "team_abr",
        "G": "g", "GS": "gs", "MP": "mp",
        "PTS": "pts", "TRB": "reb", "AST": "ast",
        "STL": "stl", "BLK": "blk", "TOV": "tov", "PF": "pf",
        "FG": "fg", "FG%": "fg_pct", "3P": "fg3", "3P%": "fg3_pct",
        "FT": "ft", "FT%": "ft_pct",
        # Advanced only
        "PER": "per", "TS%": "ts_pct", "USG%": "usg_pct",
        "WS": "ws", "BPM": "bpm", "VORP": "vorp",
    }
    df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})
    df["season"] = season
    df["stat_type"] = stat_type

    # Keep only TOT row for players traded mid-season
    if "team_abr" in df.columns:
        df = df[df["team_abr"] != "TOT"].copy()  # or keep TOT — your call

    with Session(engine) as session:
        for _, row in df.iterrows():
            # Upsert player
            player_name = row.get("name", "")
            if not player_name:
                continue
            result = session.execute(
                text("INSERT INTO players (name) VALUES (:n) "
                     "ON CONFLICT DO NOTHING RETURNING id"),
                {"n": player_name}
            ).fetchone()
            if result:
                player_id = result[0]
            else:
                player_id = session.execute(
                    text("SELECT id FROM players WHERE name = :n"),
                    {"n": player_name}
                ).scalar()

            # Build stat row
            stat_cols = [
                "g","gs","mp","pts","reb","ast","stl","blk","tov","pf",
                "fg","fg_pct","fg3","fg3_pct","ft","ft_pct",
                "per","ts_pct","usg_pct","ws","bpm","vorp"
            ]
            stat_vals = {c: float(row[c]) if c in row and pd.notna(row[c]) else None
                         for c in stat_cols}

            session.execute(
                text("""
                    INSERT INTO player_season_stats
                        (player_id, team_abr, season, stat_type,
                         g,gs,mp,pts,reb,ast,stl,blk,tov,pf,
                         fg,fg_pct,fg3,fg3_pct,ft,ft_pct,
                         per,ts_pct,usg_pct,ws,bpm,vorp)
                    VALUES
                        (:player_id, :team_abr, :season, :stat_type,
                         :g,:gs,:mp,:pts,:reb,:ast,:stl,:blk,:tov,:pf,
                         :fg,:fg_pct,:fg3,:fg3_pct,:ft,:ft_pct,
                         :per,:ts_pct,:usg_pct,:ws,:bpm,:vorp)
                    ON CONFLICT (player_id, season, team_abr, stat_type)
                    DO NOTHING
                """),
                {"player_id": player_id,
                 "team_abr": row.get("team_abr", ""),
                 "season": season, "stat_type": stat_type,
                 **stat_vals}
            )
        session.commit()


def upsert_team_standings(df: pd.DataFrame, season: int):
    engine = get_engine()
    COL_MAP = {
        "Team": "team_abr", "W": "w", "L": "l", "W/L%": "w_pct",
        "GB": "gb", "PS/G": "ps_g", "PA/G": "pa_g",
    }
    df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})
    df["season"] = season

    with Session(engine) as session:
        for _, row in df.iterrows():
            session.execute(
                text("""
                    INSERT INTO team_season_stats
                        (team_abr, season, conference, w, l, w_pct, gb, ps_g, pa_g)
                    VALUES
                        (:team_abr, :season, :conf, :w, :l, :w_pct, :gb, :ps_g, :pa_g)
                    ON CONFLICT (team_abr, season) DO NOTHING
                """),
                {
                    "team_abr": row.get("team_abr", ""),
                    "season": season,
                    "conf": row.get("conference", None),
                    "w": int(row["w"]) if "w" in row and pd.notna(row["w"]) else None,
                    "l": int(row["l"]) if "l" in row and pd.notna(row["l"]) else None,
                    "w_pct": float(row["w_pct"]) if "w_pct" in row and pd.notna(row["w_pct"]) else None,
                    "gb": None,   # GB is often '--' for first place — handle separately
                    "ps_g": float(row["ps_g"]) if "ps_g" in row and pd.notna(row["ps_g"]) else None,
                    "pa_g": float(row["pa_g"]) if "pa_g" in row and pd.notna(row["pa_g"]) else None,
                }
            )
        session.commit()


def upsert_games(df: pd.DataFrame):
    engine = get_engine()
    with Session(engine) as session:
        for _, row in df.iterrows():
            session.execute(
                text("""
                    INSERT INTO games
                        (season, game_date, home_team, away_team,
                         home_score, away_score, home_win)
                    VALUES
                        (:season, :game_date, :home_team, :away_team,
                         :home_score, :away_score, :home_win)
                    ON CONFLICT (season, game_date, home_team, away_team)
                    DO NOTHING
                """),
                {
                    "season": int(row["season"]),
                    "game_date": row["game_date"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "home_score": int(row["home_score"]) if pd.notna(row["home_score"]) else None,
                    "away_score": int(row["away_score"]) if pd.notna(row["away_score"]) else None,
                    "home_win": bool(row["home_win"]) if pd.notna(row["home_win"]) else None,
                }
            )
        session.commit()
```

---

## Step 7 — Pipeline Orchestrator (`etl/pipeline.py`)

This is the script you run once to populate the entire database.
It scrapes every season, loads it, and checkpoints progress so it resumes
if interrupted (45 seasons × multiple table types = ~3–4 hours at polite rate limits).

```python
# etl/pipeline.py
"""
Full ETL pipeline: scrape 1980–2025, load into PostgreSQL.

Usage:
    python etl/pipeline.py               # full run
    python etl/pipeline.py --start 2020  # resume from a specific season
    python etl/pipeline.py --season 2024 # single season only
"""
import argparse
import json
import time
from pathlib import Path

from scraper import (
    get_player_per_game, get_player_totals, get_player_advanced,
    get_team_standings_both, get_season_schedule,
)
from loader import upsert_player_stats, upsert_team_standings, upsert_games

CHECKPOINT_FILE = Path(".etl_checkpoint.json")
SEASONS = list(range(1980, 2026))   # 1980 = first season of this dataset


def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        return set(json.loads(CHECKPOINT_FILE.read_text()))
    return set()


def save_checkpoint(done: set):
    CHECKPOINT_FILE.write_text(json.dumps(sorted(done)))


def process_season(year: int):
    print(f"\n{'='*50}")
    print(f"  Season {year-1}–{str(year)[2:]}  ({year})")
    print(f"{'='*50}")

    # Per-game stats
    print("  → per_game...")
    try:
        df = get_player_per_game(year)
        upsert_player_stats(df, season=year, stat_type="per_game")
    except Exception as e:
        print(f"    ERROR per_game: {e}")

    # Totals
    print("  → totals...")
    try:
        df = get_player_totals(year)
        upsert_player_stats(df, season=year, stat_type="totals")
    except Exception as e:
        print(f"    ERROR totals: {e}")

    # Advanced (not available before ~1974, will just fail gracefully)
    print("  → advanced...")
    try:
        df = get_player_advanced(year)
        upsert_player_stats(df, season=year, stat_type="advanced")
    except Exception as e:
        print(f"    ERROR advanced: {e}")

    # Team standings
    print("  → standings...")
    try:
        df = get_team_standings_both(year)
        upsert_team_standings(df, season=year)
    except Exception as e:
        print(f"    ERROR standings: {e}")

    # Schedule / game results
    print("  → schedule...")
    try:
        df = get_season_schedule(year)
        if not df.empty:
            upsert_games(df)
    except Exception as e:
        print(f"    ERROR schedule: {e}")

    print(f"  ✓ Season {year} complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=None,
                        help="Start from this season year (skip earlier seasons)")
    parser.add_argument("--season", type=int, default=None,
                        help="Process a single season only")
    args = parser.parse_args()

    if args.season:
        process_season(args.season)
        return

    done = load_checkpoint()
    seasons = SEASONS if not args.start else [y for y in SEASONS if y >= args.start]

    for year in seasons:
        if year in done:
            print(f"  Season {year} already done — skipping")
            continue
        process_season(year)
        done.add(year)
        save_checkpoint(done)

    print("\n✅ ETL pipeline complete.")


if __name__ == "__main__":
    main()
```

---

## Step 8 — Running It

```bash
# 1. Apply schema
psql $DATABASE_URL -f db/schema.sql

# 2. Test with a single recent season first
python etl/pipeline.py --season 2024

# 3. Verify data landed correctly
psql $DATABASE_URL -c "SELECT COUNT(*) FROM player_season_stats;"
psql $DATABASE_URL -c "SELECT * FROM player_season_stats LIMIT 5;"

# 4. If it looks right, run the full history
#    This will take 3–5 hours (polite 4–5s delay between requests)
#    It's safe to Ctrl+C and resume — checkpoint file tracks progress
python etl/pipeline.py
```

---

## Common Issues & Fixes

| Problem | Likely cause | Fix |
|---|---|---|
| `psycopg2` install fails | Missing C libs | `brew install libpq` then `pip install psycopg2-binary` |
| `Table not found` error | Basketball-Reference changed their HTML | Print `soup.find_all("table")` to see what IDs exist |
| Rate limit / 429 error | Too many requests | Increase `sleep_s` to `6.0` or `8.0` |
| Some seasons have `NaN` advanced stats | Advanced stats page didn't exist pre-1974 | Expected — loader handles `None` gracefully |
| `unique constraint` error | Re-running the same season | Normal — `ON CONFLICT DO NOTHING` silently skips duplicates |
| Schedule missing months | NBA lockout years (1999, 2012) | Normal — scraper skips missing months |

---

## Verification Queries

Once loaded, run these to confirm data quality:

```sql
-- How many player-seasons per stat type?
SELECT stat_type, COUNT(*) FROM player_season_stats GROUP BY stat_type;

-- Season range check
SELECT MIN(season), MAX(season) FROM player_season_stats;

-- Top scorers 2024 season (per game)
SELECT p.name, s.pts, s.reb, s.ast, s.team_abr
FROM player_season_stats s
JOIN players p ON p.id = s.player_id
WHERE s.season = 2024 AND s.stat_type = 'per_game'
ORDER BY s.pts DESC
LIMIT 10;

-- Games loaded
SELECT season, COUNT(*) as games FROM games GROUP BY season ORDER BY season;
```

---

## What Phase 2 Expects From This

When the ML models and Streamlit app are built, they'll connect to this database
and query it like this:

```python
# db/connection.py  (you'll create this)
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import os

def get_session() -> Session:
    engine = create_engine(os.environ["DATABASE_URL"])
    return Session(engine)

# Example query from the ML feature engineering code:
# with get_session() as session:
#     results = session.execute(
#         text("SELECT * FROM player_season_stats WHERE season >= 2010")
#     ).fetchall()
```

Once Phase 1 is done and verified, shout and we'll build Phase 2 (game predictor model).
