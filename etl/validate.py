"""Data-quality gate: run after the ETL and before training anything.

Each check is one SQL query that returns the rows breaking a rule, so a check passes when
its query returns nothing, and when it fails the offending rows are the explanation.
Every query receives :seasons (the seasons being validated) and :steps (pipeline steps).

Usage: python -m etl.validate [--seasons 2024 2025]     (exit code 1 if any check fails)
"""
import argparse
import sys

import pandas as pd
from sqlalchemy import Connection, text

from db.connection import get_engine
from etl.pipeline import FIRST_SEASON, SEASONS, STEPS

# Regular-season, non-neutral games: the ones where home-court advantage applies.
HOME_GAMES = "game_type = 'regular' AND NOT neutral AND season = ANY(:seasons)"

CHECKS = {
    # ---- completeness
    "every ETL step finished for every season": """
        SELECT s.season, st.step
        FROM unnest(CAST(:seasons AS int[])) s(season)
        CROSS JOIN unnest(CAST(:steps AS text[]), CAST(:first_seasons AS int[])) st(step, first_season)
        WHERE s.season >= st.first_season
        EXCEPT SELECT season, step FROM etl_checkpoint""",
    "every season has games, standings and players": """
        SELECT s.season FROM unnest(CAST(:seasons AS int[])) s(season)
        WHERE NOT EXISTS (SELECT 1 FROM games g WHERE g.season = s.season)
           OR NOT EXISTS (SELECT 1 FROM team_seasons t WHERE t.season = s.season)
           OR NOT EXISTS (SELECT 1 FROM player_seasons p WHERE p.season = s.season)""",

    # ---- games
    "home teams win 53-66% overall (v1's swapped scores gave 39%)": f"""
        SELECT AVG(home_win::int) AS home_win_rate FROM games WHERE {HOME_GAMES}
        HAVING AVG(home_win::int) NOT BETWEEN 0.53 AND 0.66""",
    "home teams win at least 50% every season (2021: empty arenas)": f"""
        SELECT season, AVG(home_win::int) AS home_win_rate FROM games WHERE {HOME_GAMES}
        GROUP BY season HAVING AVG(home_win::int) < 0.50 AND season <> 2021""",
    "every game has a game_type and a neutral flag": """
        SELECT game_id, game_type, neutral FROM games
        WHERE season = ANY(:seasons) AND (game_type IS NULL OR neutral IS NULL)""",
    "all 2020 bubble games are neutral": """
        SELECT game_id FROM games
        WHERE season = 2020 AND game_date >= '2020-07-30' AND NOT neutral""",
    "known results are stored correctly": """
        SELECT k.*, g.away_score AS stored_away, g.home_score AS stored_home
        FROM (VALUES ('202310240DEN', 'LAL', 'DEN', 107, 119)   -- the game v1 stored backwards
             ) k(game_id, away, home, away_score, home_score)
        JOIN games g USING (game_id)
        WHERE g.season = ANY(:seasons)
          AND (g.away_team, g.home_team, g.away_score, g.home_score)
              IS DISTINCT FROM (k.away, k.home, k.away_score, k.home_score)""",

    # ---- standings
    "standings W-L equals W-L recomputed from the games": """
        WITH results AS (
            SELECT season, home_team AS team, home_win::int AS win FROM games WHERE game_type = 'regular'
            UNION ALL
            SELECT season, away_team, (NOT home_win)::int FROM games WHERE game_type = 'regular')
        SELECT t.season, t.team_abr, t.w, t.l, SUM(r.win) AS games_w, COUNT(r.win) - SUM(r.win) AS games_l
        FROM team_seasons t LEFT JOIN results r ON r.season = t.season AND r.team = t.team_abr
        WHERE t.season = ANY(:seasons)
        GROUP BY t.season, t.team_abr, t.w, t.l
        HAVING t.w IS DISTINCT FROM SUM(r.win) OR t.l IS DISTINCT FROM COUNT(r.win) - SUM(r.win)""",
    "22-30 teams per season, in both conferences": """
        SELECT season, COUNT(*) AS teams, COUNT(DISTINCT conference) AS conferences
        FROM team_seasons WHERE season = ANY(:seasons) GROUP BY season
        HAVING COUNT(*) NOT BETWEEN 22 AND 30 OR COUNT(DISTINCT conference) <> 2""",
    "lockout seasons are short (1999: 50 games, 2012: 66)": """
        SELECT season, MAX(w + l) AS games_per_team FROM team_seasons
        WHERE season IN (1999, 2012) AND season = ANY(:seasons) GROUP BY season
        HAVING MAX(w + l) <> CASE season WHEN 1999 THEN 50 ELSE 66 END""",

    # ---- players
    "no player row has an unknown team": """
        SELECT season, team_abr, COUNT(*) FROM player_seasons
        WHERE team_abr = 'UNK' AND season = ANY(:seasons) GROUP BY 1, 2""",
    "250-700 players per season": """
        SELECT season, COUNT(*) AS players FROM player_seasons
        WHERE is_primary AND season = ANY(:seasons) GROUP BY season
        HAVING COUNT(*) NOT BETWEEN 250 AND 700""",
    "exactly one primary row per player-season": """
        SELECT player_id, season, COUNT(*) FROM player_seasons
        WHERE is_primary AND season = ANY(:seasons) GROUP BY 1, 2 HAVING COUNT(*) > 1""",
    "advanced stats present for every player who played": """
        SELECT player_id, season, team_abr FROM player_seasons
        WHERE season = ANY(:seasons) AND g > 0 AND mp_total IS NULL""",
    "a traded player's TOT games equal the sum of their per-team games": """
        SELECT tot.player_id, tot.season, tot.g, SUM(t.g) AS per_team_g
        FROM player_seasons tot
        JOIN player_seasons t ON t.player_id = tot.player_id AND t.season = tot.season
                             AND t.team_abr <> 'TOT'
        WHERE tot.team_abr = 'TOT' AND tot.season = ANY(:seasons)
        GROUP BY tot.player_id, tot.season, tot.g HAVING tot.g <> SUM(t.g)""",
    # Box totals: sum(player PPG x games) over a team's rows = team PS/G x games, within 1%
    # (per-game averages are rounded to 0.1, which keeps the real error under 0.3%).
    "player points add up to team points": """
        WITH p AS (SELECT season, team_abr, SUM(pts * g) AS player_pts FROM player_seasons
                   WHERE team_abr <> 'TOT' GROUP BY 1, 2)
        SELECT t.season, t.team_abr, p.player_pts, t.ps_g * (t.w + t.l) AS team_pts
        FROM team_seasons t LEFT JOIN p USING (season, team_abr)
        WHERE t.season = ANY(:seasons)
          AND (p.player_pts IS NULL OR ABS(p.player_pts / (t.ps_g * (t.w + t.l)) - 1) > 0.01)""",

    # ---- money: betting lines, payroll, cap (1994 onward)
    # 1995-96 archive pages list only some teams; 1999 (lockout) has no lines at all
    "every team has a preseason betting line (1994+, except 1995, 1996, 1999)": """
        SELECT t.season, t.team_abr FROM team_seasons t
        LEFT JOIN win_totals w USING (season, team_abr)
        WHERE t.season >= 1994 AND t.season NOT IN (1995, 1996, 1999) AND t.season = ANY(:seasons)
          AND w.line IS NULL""",
    "the betting site's actual wins equal our standings": """
        SELECT w.season, w.team_abr, w.actual_wins, t.w FROM win_totals w
        JOIN team_seasons t USING (season, team_abr)
        WHERE w.season = ANY(:seasons) AND w.actual_wins IS DISTINCT FROM t.w""",
    "every team-season since 1994 has a payroll and a salary cap": """
        SELECT t.season, t.team_abr, p.payroll, p.cap FROM team_seasons t
        LEFT JOIN team_payrolls p USING (season, team_abr)
        WHERE t.season >= 1994 AND t.season = ANY(:seasons) AND (p.payroll IS NULL OR p.cap IS NULL)""",
    # real extremes: 2006 Knicks 2.56x and 2003 "Jail Blazers" 2.62x the cap
    "payrolls are plausible (35-300% of the cap)": """
        SELECT season, team_abr, payroll, cap, ROUND(payroll_to_cap::numeric, 2) AS ratio FROM team_payrolls
        WHERE season = ANY(:seasons) AND payroll_to_cap NOT BETWEEN 0.35 AND 3.0""",

    # ---- franchises
    "30 franchises, all active in the latest season": """
        SELECT franchise_id, last_season FROM franchises
        WHERE last_season < (SELECT MAX(season) FROM team_seasons)
        UNION ALL
        SELECT 'franchise count = ' || COUNT(*), NULL FROM franchises HAVING COUNT(*) <> 30""",
    "every retired abbreviation belongs to a current franchise (etl/teams.py)": """
        SELECT team_abr, last_season FROM teams
        WHERE franchise_id = team_abr AND last_season < (SELECT MAX(season) FROM team_seasons)""",
}


def run_checks(conn: Connection, seasons: list[int]) -> dict[str, pd.DataFrame]:
    """{check name: offending rows}; a check passed when its DataFrame is empty."""
    params = {"seasons": seasons, "steps": list(STEPS),
              "first_seasons": [FIRST_SEASON.get(step, SEASONS[0]) for step in STEPS]}
    return {name: pd.read_sql_query(text(sql), conn, params=params) for name, sql in CHECKS.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the NBA database.")
    parser.add_argument("--seasons", nargs="+", type=int, default=SEASONS,
                        help="seasons to check (default: all, 1980 to the last completed one)")
    args = parser.parse_args()

    print(f"Validating {len(args.seasons)} season(s): {args.seasons[0]}–{args.seasons[-1]}\n")
    with get_engine().connect() as conn:
        results = run_checks(conn, args.seasons)
    for name, bad in results.items():
        print(f"  {'✓' if bad.empty else '✗'} {name}")
        if not bad.empty:
            print("      " + bad.head(5).to_string(index=False).replace("\n", "\n      "))
            if len(bad) > 5:
                print(f"      … {len(bad) - 5} more")

    failed = sum(not bad.empty for bad in results.values())
    print(f"\n{'All checks passed.' if not failed else f'{failed} check(s) FAILED.'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
