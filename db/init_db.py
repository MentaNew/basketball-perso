"""Create all tables (idempotent).  Usage: python -m db.init_db [--drop]"""
import argparse

from sqlalchemy import text

from db.connection import get_engine
from db.models import Base

# Columns added after the first release; create_all() doesn't alter existing tables.
MIGRATIONS = [
    "ALTER TABLE games ADD COLUMN IF NOT EXISTS neutral BOOLEAN",
]

# Lookup views derived from team_seasons, so they can never drift out of sync with it.
VIEWS = [
    # one row per BRef abbreviation: which franchise, its latest name, when it was used
    """CREATE OR REPLACE VIEW teams AS
       SELECT team_abr, franchise_id,
              (array_agg(full_name ORDER BY season DESC))[1] AS full_name,
              MIN(season) AS first_season, MAX(season) AS last_season
       FROM team_seasons GROUP BY team_abr, franchise_id""",
    # one row per franchise, named after its most recent identity
    """CREATE OR REPLACE VIEW franchises AS
       SELECT DISTINCT ON (franchise_id) franchise_id, team_abr AS current_abr, full_name,
              MIN(first_season) OVER w AS first_season, MAX(last_season) OVER w AS last_season,
              COUNT(*) OVER w AS n_abbreviations
       FROM teams WINDOW w AS (PARTITION BY franchise_id)
       ORDER BY franchise_id, teams.last_season DESC  -- the source column, not the window alias""",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true", help="drop all tables first (destroys data)")
    args = parser.parse_args()
    engine = get_engine()
    if args.drop:
        with engine.begin() as conn:
            conn.execute(text("DROP VIEW IF EXISTS franchises, teams"))
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for stmt in MIGRATIONS + VIEWS:
            conn.execute(text(stmt))
    print("Tables:", ", ".join(Base.metadata.tables), "| views: teams, franchises")


if __name__ == "__main__":
    main()
