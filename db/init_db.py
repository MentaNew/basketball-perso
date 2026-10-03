"""Create all tables (idempotent).  Usage: python -m db.init_db [--drop]"""
import argparse

from db.connection import get_engine
from db.models import Base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true", help="drop all tables first (destroys data)")
    args = parser.parse_args()
    engine = get_engine()
    if args.drop:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    print("Tables:", ", ".join(Base.metadata.tables))


if __name__ == "__main__":
    main()
