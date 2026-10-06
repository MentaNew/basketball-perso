"""Create the schema: tables from db/models.py, plus two lookup views.

Usage:  python -m db.init_db          # create what's missing (safe to re-run)
        python -m db.init_db --drop   # wipe everything first (destroys data)

create_all() only creates missing tables; it never alters an existing one. Changing a
column on a populated DB therefore needs a migration tool (Alembic, coming in level L1).
"""
import argparse

from sqlalchemy import Engine, text

from db.connection import get_engine
from db.models import Base

# The views are derived from team_seasons, so they can never disagree with it.
VIEWS = {
    # One row per abbreviation: its franchise, latest name and the seasons it was used.
    "teams": """
        SELECT team_abr, franchise_id,
               (array_agg(full_name ORDER BY season DESC))[1] AS full_name,
               MIN(season) AS first_season, MAX(season) AS last_season
        FROM team_seasons
        GROUP BY team_abr, franchise_id""",
    # One row per franchise, named after its current identity (OKC, not SEA).
    "franchises": """
        SELECT DISTINCT ON (franchise_id)
               franchise_id, team_abr AS current_abr, full_name,
               MIN(first_season) OVER w AS first_season,
               MAX(last_season) OVER w AS last_season,
               COUNT(*) OVER w AS n_abbreviations
        FROM teams
        WINDOW w AS (PARTITION BY franchise_id)
        -- teams.last_season: the source column; a bare last_season would mean the window result
        ORDER BY franchise_id, teams.last_season DESC""",
}


def init_db(engine: Engine, drop: bool = False) -> None:
    with engine.begin() as conn:
        if drop:
            # views depend on the tables, so they go first (in reverse creation order)
            for name in reversed(VIEWS):
                conn.execute(text(f"DROP VIEW IF EXISTS {name}"))
            Base.metadata.drop_all(conn)
        Base.metadata.create_all(conn)
        for name, query in VIEWS.items():
            conn.execute(text(f"CREATE OR REPLACE VIEW {name} AS {query}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drop", action="store_true", help="drop all tables first (destroys data)")
    args = parser.parse_args()
    init_db(get_engine(), drop=args.drop)
    print(f"Tables: {', '.join(Base.metadata.tables)} | views: {', '.join(VIEWS)}")


if __name__ == "__main__":
    main()
