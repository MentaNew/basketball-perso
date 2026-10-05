"""Data-quality gate: run after the ETL, before training anything.

Usage: python -m etl.validate [--seasons 2024 2025]
Exit code 1 if any check fails.
"""
import argparse
import sys

from db.connection import query_df
from etl.pipeline import SEASONS, STEPS

# Known results used as spot checks: (game_id, away, home, away_score, home_score).
# Add more as you verify them against box scores.
KNOWN_GAMES = [
    ("202310240DEN", "LAL", "DEN", 107, 119),  # the game stored inverted in v1 (§7.1)
]
LOCKOUT_SEASONS = {1999: 50, 2012: 66}  # games per team


class Report:
    def __init__(self):
        self.failures = 0

    def check(self, ok: bool, label: str, detail: str = ""):
        self.failures += not ok
        print(f"  {'✓' if ok else '✗'} {label}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, help="limit per-season checks")
    args = ap.parse_args()
    r = Report()

    loaded = query_df("SELECT DISTINCT season FROM games ORDER BY 1")["season"].tolist()
    seasons = [s for s in (args.seasons or loaded) if s in loaded]
    sql_seasons = {"seasons": seasons}
    print(f"Seasons with games: {len(loaded)} ({loaded[:1]}..{loaded[-1:]})")

    print("\nCompleteness")
    expected = args.seasons or SEASONS
    done = query_df("SELECT season, step FROM etl_checkpoint")
    done = set(zip(done.season, done.step, strict=True))
    missing = [f"{s}:{st}" for s in expected for st in STEPS if (s, st) not in done]
    r.check(not missing, f"all {len(STEPS)} ETL steps checkpointed for {len(expected)} seasons "
            f"({expected[0]}–{expected[-1]})", ", ".join(missing[:8]) + (" …" if len(missing) > 8 else ""))
    present = query_df("""SELECT s.season,
                                 EXISTS (SELECT 1 FROM games g WHERE g.season = s.season) games,
                                 EXISTS (SELECT 1 FROM team_seasons t WHERE t.season = s.season) teams,
                                 EXISTS (SELECT 1 FROM player_seasons p WHERE p.season = s.season) players
                          FROM unnest(CAST(:seasons AS int[])) AS s(season)""", {"seasons": list(expected)})
    empty = present[~(present.games & present.teams & present.players)]
    r.check(empty.empty, "every season has games, standings and players",
            ", ".join(map(str, empty.season)))

    print("\nGames")
    home_games = "game_type = 'regular' AND NOT neutral AND season = ANY(:seasons)"
    hw = query_df(f"""SELECT season, AVG(home_win::int) AS hw, COUNT(*) AS n
                      FROM games WHERE {home_games} GROUP BY 1 ORDER BY 1""", sql_seasons)
    overall = query_df(f"SELECT AVG(home_win::int) FROM games WHERE {home_games}",
                       sql_seasons).iat[0, 0]
    r.check(0.53 <= overall <= 0.66, "overall home-win rate (regular, non-neutral) in [0.53, 0.66]",
            f"{overall:.3f}")
    # 2021 was played in near-empty arenas
    bad = hw[(hw.hw < 0.50) & (hw.season != 2021)]
    r.check(bad.empty, "every season home-win rate ≥ 0.50 (2021 excepted)",
            ", ".join(f"{s}:{v:.2f}" for s, v in zip(bad.season, bad.hw, strict=True)))
    unflagged = query_df("SELECT COUNT(*) FROM games WHERE neutral IS NULL "
                         "AND season = ANY(:seasons)", sql_seasons).iat[0, 0]
    r.check(unflagged == 0, "every game has a neutral flag", f"{unflagged} unflagged")
    if 2020 in seasons:
        bubble = query_df("""SELECT COUNT(*) FILTER (WHERE neutral) n, COUNT(*) total FROM games
                             WHERE season = 2020 AND game_date >= '2020-07-30'""").iloc[0]
        r.check(bubble.n == bubble.total > 0, "2020 bubble games flagged neutral",
                f"{bubble.n}/{bubble.total}")
    untyped = query_df("SELECT COUNT(*) FROM games WHERE game_type IS NULL "
                       "AND season = ANY(:seasons)", sql_seasons).iat[0, 0]
    r.check(untyped == 0, "every game has a game_type", f"{untyped} untyped")

    print("\nStandings vs. games")
    cmp = query_df("""
        WITH g AS (
            SELECT season, team, SUM(win) w, COUNT(*) - SUM(win) l FROM (
                SELECT season, home_team team, home_win::int win FROM games WHERE game_type='regular'
                UNION ALL
                SELECT season, away_team, (NOT home_win)::int FROM games WHERE game_type='regular'
            ) x GROUP BY 1, 2)
        SELECT t.season, t.team_abr, t.w, t.l, g.w gw, g.l gl
        FROM team_seasons t LEFT JOIN g ON g.season = t.season AND g.team = t.team_abr
        WHERE t.season = ANY(:seasons)""", sql_seasons)
    mismatch = cmp[(cmp.w != cmp.gw) | (cmp.l != cmp.gl)]
    r.check(mismatch.empty, "standings W/L == W/L recomputed from games",
            mismatch.head(5).to_dict("records") if not mismatch.empty else "")
    per = query_df("""SELECT season, COUNT(*) n, COUNT(DISTINCT conference) confs
                      FROM team_seasons WHERE season = ANY(:seasons) GROUP BY 1""", sql_seasons)
    bad = per[(per.n < 22) | (per.n > 30) | (per.confs != 2)]
    r.check(bad.empty and len(per) == len(seasons),
            "each season has 22–30 teams across both conferences",
            bad.to_dict("records") if not bad.empty else "")
    for season, gpt in LOCKOUT_SEASONS.items():
        if season in seasons:
            n = query_df("SELECT MAX(w + l) FROM team_seasons WHERE season = :s",
                         {"s": season}).iat[0, 0]
            r.check(n == gpt, f"{season} lockout season: {gpt} games per team", f"got {n}")

    print("\nPlayers")
    unk = query_df("SELECT COUNT(*) FROM player_seasons WHERE team_abr = 'UNK'").iat[0, 0]
    r.check(unk == 0, "no player rows with team 'UNK'", f"{unk} rows")
    nulls = query_df("SELECT COUNT(*) FROM players WHERE bref_id IS NULL").iat[0, 0]
    r.check(nulls == 0, "every player has a bref_id")
    counts = query_df("""SELECT season, COUNT(*) n FROM player_seasons
                         WHERE is_primary AND season = ANY(:seasons) GROUP BY 1""", sql_seasons)
    bad = counts[(counts.n < 250) | (counts.n > 700)]
    r.check(bad.empty, "250–700 primary player rows per season",
            bad.to_dict("records") if not bad.empty else "")
    dup = query_df("""SELECT COUNT(*) FROM (SELECT player_id, season FROM player_seasons
                      WHERE is_primary GROUP BY 1, 2 HAVING COUNT(*) > 1) x""").iat[0, 0]
    r.check(dup == 0, "exactly one primary row per player-season", f"{dup} duplicates")
    no_adv = query_df("""SELECT COUNT(*) FROM player_seasons
                         WHERE season = ANY(:seasons) AND mp_total IS NULL AND g > 0""",
                      sql_seasons).iat[0, 0]
    r.check(no_adv == 0, "advanced stats present wherever minutes were played", f"{no_adv} rows")

    print("\nBox totals")
    # Σ (player PPG × games) over a team's per-team rows must equal PS/G × games from the
    # standings.  Rounding of per-game averages keeps the real error under 0.3%.
    box = query_df("""
        WITH p AS (SELECT season, team_abr, SUM(pts * g) pts FROM player_seasons
                   WHERE team_abr <> 'TOT' AND season = ANY(:seasons) GROUP BY 1, 2)
        SELECT t.season, t.team_abr, p.pts AS player_pts, t.ps_g * (t.w + t.l) AS team_pts
        FROM team_seasons t LEFT JOIN p USING (season, team_abr)
        WHERE t.season = ANY(:seasons)""", sql_seasons)
    box["err"] = (box.player_pts / box.team_pts - 1).abs()
    off = box[box.err.isna() | (box.err > 0.01)]
    r.check(off.empty, "player points add up to team points (±1%) for every team-season",
            f"max err {box.err.max():.2%}" if off.empty else off.head(5).to_dict("records"))
    multi = query_df("""SELECT COUNT(*) FROM player_seasons t
                        WHERE t.team_abr = 'TOT' AND season = ANY(:seasons) AND t.g <> (
                            SELECT SUM(x.g) FROM player_seasons x WHERE x.player_id = t.player_id
                            AND x.season = t.season AND x.team_abr <> 'TOT')""", sql_seasons).iat[0, 0]
    r.check(multi == 0, "traded players: TOT games == sum of per-team games", f"{multi} mismatches")

    print("\nFranchises")
    fr = query_df("SELECT * FROM franchises")
    r.check(len(fr) == 30 and (fr.last_season == fr.last_season.max()).all(),
            "30 franchises, all active in the latest season", f"{len(fr)} franchises")
    unmapped = query_df("""SELECT team_abr FROM teams WHERE last_season < (SELECT MAX(season) FROM games)
                           AND franchise_id = team_abr""")
    r.check(unmapped.empty, "every retired abbreviation maps to a current franchise (etl/teams.py)",
            ", ".join(unmapped.team_abr))

    print("\nSpot checks")
    for gid, away, home, a_pts, h_pts in KNOWN_GAMES:
        g = query_df("SELECT * FROM games WHERE game_id = :g", {"g": gid})
        if g.empty:
            print(f"  · {gid} not loaded (season not scraped yet)")
            continue
        row = g.iloc[0]
        r.check((row.away_team, row.home_team, row.away_score, row.home_score)
                == (away, home, a_pts, h_pts), f"{gid}: {away} {a_pts} @ {home} {h_pts}",
                f"got {row.away_team} {row.away_score} @ {row.home_team} {row.home_score}")

    print(f"\n{'All checks passed.' if not r.failures else f'{r.failures} check(s) FAILED.'}")
    return 1 if r.failures else 0


if __name__ == "__main__":
    sys.exit(main())
