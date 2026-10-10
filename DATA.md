# Data Catalogue

Everything this project stores: what it is, where it lives, how it's made, and how far to trust
it. Audited 2026-10-10.

## Where things live

| What | Location | Size | Rebuild with |
|---|---|---|---|
| **Database** (active) | PostgreSQL `nba_platform_l0` (`DATABASE_URL` in `.env`) | ~34 MB | `python -m db.init_db && python -m etl.pipeline --force` |
| Reference DB (frozen) | `nba_platform_v2`: identical copy built by the pre-rewrite code | ~47 MB | don't write to it |
| Test DB | `nba_platform_test` (`TEST_DATABASE_URL`), wiped by `tests/test_loader.py` | tiny | automatic |
| Old v1 DB | `nba_platform`: v1 data with known bugs | — | don't use |
| **Raw HTML cache** | `data/raw/` (git-ignored) | ~350 MB | `python -m etl.pipeline --refresh` (slow, ~1 h) |
| Trained models | `models/*.pkl`, `*.pt` (git-ignored) | ~4 MB | `python -m ml.<model> train` |
| Test fixtures | `tests/fixtures/*.html.gz` (in git) | ~1 MB | saved by hand from real pages |

The raw cache is the real source of truth. The database can always be rebuilt from it in
~1.5 minutes without touching the network.

## Database tables (`db/models.py`)

`season` = the ending year (2026 = 2025-26). Teams are abbreviations *as of that season*.

| Table | Rows | Seasons | Key | What |
|---|---|---|---|---|
| `players` | 3,819 | — | `bref_id` | name, accent-free `search_name`, approximate `birth_year` |
| `player_seasons` | 25,349 | 1980–2026 | (player, season, team) | per-game + advanced stats in one wide row. Traded players: one row per team + a `TOT` row; `is_primary` marks the one to analyse |
| `team_seasons` | 1,314 | 1980–2026 | (team, season) | final standings: W-L, conference/division, PS/G, PA/G, SRS, playoffs, `franchise_id` |
| `games` | 56,454 | 1980–2026 | `game_id` | every played game: teams, scores, `game_type` (52,848 regular, 3,566 playoff, 37 play-in, 3 Cup finals), `neutral`, remarks |
| `etl_checkpoint` | 141 | 1980–2026 | (season, step) | which pipeline steps finished |
| `teams` *(view)* | 40 | — | `team_abr` | abbreviation → franchise, name, years used |
| `franchises` *(view)* | 30 | — | `franchise_id` | current identity of each franchise (SEA → OKC, NJN → BRK …) |

## Raw HTML cache (`data/raw/`)

| Path | Source | Coverage | Loaded into DB? |
|---|---|---|---|
| `{season}/per_game.html`, `advanced.html` | basketball-reference.com league pages | 1980–2026 | ✅ `player_seasons` |
| `{season}/standings.html` | BRef standings | 1980–2026 | ✅ `team_seasons` |
| `{season}/games.html`, `games-{month}.html` | BRef schedule pages (`.404` files mark months that don't exist) | 1980–2026 | ✅ `games` |
| `{season}/team_{ABR}.html` | BRef team-season pages: **salaries** (in a hidden HTML comment) and roster | 1994–2026 | ❌ roadmap A0 (`team_payrolls`) |
| `odds/win_totals_{season}.html` | covers.com/sportsoddshistory: preseason win totals with line, odds, actual wins | 1994–2026 (no 1999 lockout; the archive has no lines for an unfinished season, so 2026-27 needs another source) | ❌ roadmap A0 (`win_totals`) |

Downloading rules: BRef ~20 req/min (we use 4–5 s per request); sportsoddshistory asks for a
3 s crawl delay. HoopsHype (payrolls, HTTP 402) and 2kratings.com (403) block scripts, so we
don't use them.

## Quality: what's checked, and the known quirks

`python -m etl.validate` runs 18 SQL checks (see `PHASE1_ETL_GUIDE.md`, Step 8) and passes on
all 47 seasons. `tests/test_validate.py` proves the checks catch re-injected past bugs.

An extra audit on 2026-10-10 found no errors. Every anomaly is a real event:

| Looks odd | Explanation |
|---|---|
| `fg3_pct` missing for 13.9% of rows, `ft_pct` for 3.3% | players with zero attempts (common in the 1980s), so no percentage exists |
| 1980–83 have only 43–48 playoff games | 12-team playoffs with best-of-3 first rounds |
| 2013: two teams played 81 games | BOS–IND cancelled after the Boston Marathon bombing |
| 2020: 64–75 games per team | COVID suspension; the bubble restart is flagged `neutral` |
| CHI scored 49 on 1999-04-10 | real: the lowest total of the shot-clock era |
| 25 names shared by different players | handled: identity is `bref_id`, never the name |

**Derived or approximate fields:**
- `game_type` is *inferred*: the first W+L games of each team are regular season; play-in comes
  from the remarks; the Cup final is the last "NBA Cup"/"In-Season Tournament" game.
- `neutral` covers games abroad plus the 2020 bubble (domestic alternate arenas count as home).
- `birth_year` = season − age − 1, so it can be off by one.
- `franchise_id` comes from the hand-written map in `etl/teams.py`.

**Live market data (not stored yet):** Kalshi and Polymarket public APIs. See
`SIDE_PROJECTS.md` for what they offer and the snapshot plan.

**Not available (yet):**
- opening-night rosters (`player_seasons` knows who *ended up* playing for each team);
- injuries;
- player game logs;
- shot locations;
- historical 2K ratings.
