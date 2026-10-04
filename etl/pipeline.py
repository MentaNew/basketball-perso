"""Full ETL: scrape 1980 through the last completed season from Basketball-Reference into PostgreSQL.

Progress is checkpointed per (season, step) in the `etl_checkpoint` table, and only after
the step succeeds, so a failed step is retried on the next run.

Usage:
    python -m etl.pipeline                    # everything not yet done
    python -m etl.pipeline --season 2024      # one season
    python -m etl.pipeline --start 2000 --end 2010
    python -m etl.pipeline --season 2024 --force     # re-parse from the HTML cache
    python -m etl.pipeline --season 2024 --refresh   # re-download pages too
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
# standings must precede schedule: game-type classification needs each team's W+L
STEPS = ["per_game", "advanced", "standings", "schedule"]


def run_step(season: int, step: str, refresh: bool) -> int:
    if step == "per_game":
        return loader.upsert_player_seasons(scraper.get_player_per_game(season, refresh), season, step)
    if step == "advanced":
        return loader.upsert_player_seasons(scraper.get_player_advanced(season, refresh), season, step)
    if step == "standings":
        return loader.upsert_standings(scraper.get_standings(season, refresh), season)
    if step == "schedule":
        return loader.upsert_games(scraper.get_season_schedule(season, refresh), season)
    raise ValueError(step)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--start", type=int, default=SEASONS[0])
    ap.add_argument("--end", type=int, default=SEASONS[-1])
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    ap.add_argument("--force", action="store_true", help="ignore checkpoint (uses HTML cache)")
    ap.add_argument("--refresh", action="store_true", help="re-download pages (implies --force)")
    args = ap.parse_args()

    seasons = [args.season] if args.season else [s for s in SEASONS if args.start <= s <= args.end]
    done = set() if (args.force or args.refresh) else loader.completed_steps()
    failures = []

    for season in seasons:
        todo = [s for s in args.steps if (season, s) not in done]
        if not todo:
            continue
        print(f"── {season - 1}-{str(season)[2:]} ({season})", flush=True)
        for step in todo:
            try:
                n = run_step(season, step, args.refresh)
                loader.mark_done(season, step, n)
                print(f"   ✓ {step:<9} {n:>6} rows", flush=True)
            except scraper.RateLimited as e:
                print(f"   ✗ {e}", flush=True)
                return 2
            except Exception as e:  # keep going; the step stays un-checkpointed
                failures.append((season, step, repr(e)))
                print(f"   ✗ {step:<9} {e!r}", flush=True)
                traceback.print_exc(limit=2, file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} step(s) failed; re-run to retry:")
        for f in failures:
            print("  ", *f)
        return 1
    print("\nETL complete. Run `python -m etl.validate` next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
