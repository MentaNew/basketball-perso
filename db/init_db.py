"""Create all tables (idempotent).  Usage: python -m db.init_db [--drop]"""
import argparse

from sqlalchemy import text

from db.connection import get_engine
from db.models import Base

# Columns added after the first release; create_all() doesn't alter existing tables.
MIGRATIONS = [
    "ALTER TABLE games ADD COLUMN IF NOT EXISTS neutral BOOLEAN",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true", help="drop all tables first (destroys data)")
    args = parser.parse_args()
    engine = get_engine()
    if args.drop:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for stmt in MIGRATIONS:
            conn.execute(text(stmt))
    print("Tables:", ", ".join(Base.metadata.tables))


if __name__ == "__main__":
    main()
