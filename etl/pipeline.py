"""ETL command line: scrape seasons from Basketball-Reference into PostgreSQL.

Each season runs these steps, in this order (game types and team names need the standings):

    players -> standings -> schedule -> win_totals -> salaries      (the last two from 1994)

A step is recorded in the etl_checkpoint table only after it succeeds, so an interrupted
or failed run simply resumes where it stopped on the next run.

Usage:
    python -m etl.pipeline                         # every (season, step) not done yet
    python -m etl.pipeline --season 2026           # one season
    python -m etl.pipeline --start 2000 --end 2010
    python -m etl.pipeline --steps schedule        # only some steps
    python -m etl.pipeline --force                 # redo everything from the HTML cache
    python -m etl.pipeline --refresh               # redo everything, re-downloading pages
"""
import argparse
import sys
import traceback
from datetime import date

from etl import loader, scraper


def last_completed_season(today: date | None = None) -> int:
    """Season Y (ending year) is over once the Finals end, so count it from July 1 of Y."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


SEASONS = list(range(1980, last_completed_season() + 1))

STEPS = {  # step name -> (download + parse, load); dict order is the run order
    "players": (scraper.get_players, loader.load_players),
    "standings": (scraper.get_standings, loader.load_standings),
    "schedule": (scraper.get_schedule, loader.load_games),
    "win_totals": (scraper.get_win_totals, loader.load_win_totals),
    "salaries": (scraper.get_salaries, loader.load_salaries),
}
FIRST_SEASON = {"win_totals": 1994, "salaries": 1994}  # no betting-line archive before 1993-94


def applies(season: int, step: str) -> bool:
    return season >= FIRST_SEASON.get(step, SEASONS[0])


def run_step(season: int, step: str, refresh: bool) -> int:
    extract, load = STEPS[step]
    return load(extract(season, refresh), season)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape NBA seasons into PostgreSQL.")
    parser.add_argument("--season", type=int, help="one season (ending year, e.g. 2026)")
    parser.add_argument("--start", type=int, default=SEASONS[0])
    parser.add_argument("--end", type=int, default=SEASONS[-1])
    parser.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    parser.add_argument("--force", action="store_true", help="ignore the checkpoint (uses the HTML cache)")
    parser.add_argument("--refresh", action="store_true", help="re-download pages (implies --force)")
    args = parser.parse_args()

    seasons = [args.season] if args.season else list(range(args.start, args.end + 1))
    steps = [s for s in STEPS if s in args.steps]  # keep the dependency order
    done = set() if (args.force or args.refresh) else loader.completed_steps()
    failures = []

    for season in seasons:
        todo = [step for step in steps if applies(season, step) and (season, step) not in done]
        if not todo:
            continue
        print(f"── {season - 1}-{str(season)[2:]} ({season})", flush=True)
        for step in todo:
            try:
                rows = run_step(season, step, args.refresh)
            except scraper.RateLimited as error:
                print(f"   ✗ {error}", flush=True)
                return 2  # stop everything: continuing would extend the ban
            except Exception as error:  # record it, keep going with the next step
                failures.append((season, step, repr(error)))
                print(f"   ✗ {step:<9} {error!r}", flush=True)
                traceback.print_exc(limit=2, file=sys.stderr)
                continue
            loader.mark_done(season, step, rows)
            print(f"   ✓ {step:<9} {rows:>6} rows", flush=True)

    if failures:
        print(f"\n{len(failures)} step(s) failed; run again to retry them:")
        for failure in failures:
            print("  ", *failure)
        return 1
    print("\nETL complete. Next: python -m etl.validate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
