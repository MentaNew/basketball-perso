# Phase 1 — ETL Pipeline & Database

Build the data foundation: scrape every NBA season since 1979-80 from Basketball-Reference,
store it in PostgreSQL, and **prove it is correct** before any model touches it.

Files: `db/`, `etl/`, `tests/test_parsers.py`, `tests/test_game_types.py`,
`tests/test_scraper.py`, `tests/test_loader.py`, `tests/fixtures/`.

> This guide replaces the original Phase 1 guide. That version's code swapped home and away
> scores, keyed players by name, missed most standings and checkpointed failed seasons as
> done (blueprint §7.1, §7.3, §7.4). Everything below describes the rebuilt pipeline.

---

## Step 0 — PostgreSQL

```bash
brew install postgresql@17
brew services start postgresql@17
psql postgres -c "SELECT version();"
```

```sql
-- inside psql
CREATE USER nba_user WITH PASSWORD 'yourpassword';
CREATE DATABASE nba_platform_v2 OWNER nba_user;
```

```sql
-- optional: a throwaway DB for the loader tests (wiped on every test run)
CREATE DATABASE nba_platform_test OWNER nba_user;
```

`.env` (git-ignored; template in `.env.example`):

```
DATABASE_URL=postgresql://nba_user:yourpassword@localhost:5432/nba_platform_v2
TEST_DATABASE_URL=postgresql://nba_user:yourpassword@localhost:5432/nba_platform_test
```

---

## Step 1 — Python environment

```bash
conda create -n nba python=3.11 -y
conda activate nba
pip install -r requirements.txt        # pinned versions; includes pytest and ruff
export PYTHONPATH=.                    # every command runs as `python -m package.module`
```

---

## Step 2 — Structure

```
db/
  models.py       SQLAlchemy tables: the single source of truth for the schema
  connection.py   get_engine() (cached, pool_pre_ping) and query_df(sql, params)
  init_db.py      init_db(): create tables + (re)create the lookup views (idempotent)
etl/
  scraper.py      network + HTML cache. The only file that talks to the internet
  parsers.py      pure HTML → DataFrame functions. No network, no DB, fully tested
  loader.py       idempotent loads + game-type inference + checkpoint helpers
  pipeline.py     CLI orchestrator
  validate.py     data-quality gate
  teams.py        franchise map (SEA → OKC, …)
tests/
  fixtures/*.html.gz   real BRef pages used by the parser and loader tests
  test_parsers.py      HTML → DataFrame on the fixtures
  test_game_types.py   regular / play-in / playoff / Cup-final inference
  test_scraper.py      cache, 404 memory, 429 handling (network faked)
  test_loader.py       idempotency and refresh against TEST_DATABASE_URL
```

The key split is **scraper vs. parsers**. Network code is slow, flaky and rate-limited.
Parsing is where the bugs live. Keeping parsing pure means every bug can be reproduced and
tested offline against a saved page.

---

## Step 3 — Schema (`db/models.py`)

```bash
python -m db.init_db          # create tables and apply migrations; safe to re-run
```

| Table | Key | Contents |
|---|---|---|
| `players` | `id`, unique `bref_id` | `name` (HOF `*` stripped), `search_name` (lower-case, accent-free), `birth_year` (approx.) |
| `player_seasons` | unique `(player_id, season, team_abr)` | **wide** row: per-game stats + advanced stats, `age`, `pos`, `team_list`, `is_primary` |
| `team_seasons` | unique `(team_abr, season)` | `franchise_id`, `full_name`, `conference`, `division`, W/L, `ps_g`/`pa_g`, `srs`, `made_playoffs` |
| `games` | `game_id` (BRef box-score id, e.g. `202310240DEN`) | away/home abbreviation + name + score, `home_win`, `overtimes`, `attendance`, `remarks`, `game_type`, `neutral` |
| `etl_checkpoint` | `(season, step)` | `rows`, `completed_at` |
| `teams` *(view)* | `team_abr` | `franchise_id`, latest `full_name`, `first_season`, `last_season` |
| `franchises` *(view)* | `franchise_id` | `current_abr`, current `full_name`, `first_season`, `last_season`, `n_abbreviations` |

Design decisions:
- **`season` is the ending year:** `2024` means 2023-24.
- **Identity is `bref_id`** (`jamesle01`), taken from the `data-append-csv` attribute of the
  player cell. Names are not unique: two Tony Mitchells both played in 2013-14.
- **Wide player rows.** v1 stored per-game and advanced data as separate long-format rows
  that every consumer had to pivot. Now one row holds both: the `players` step reads both
  pages and merges them (`parsers.merge_player_tables`) before loading.
- **Traded players** get one row per team **plus** a combined row with `team_abr = 'TOT'`
  (BRef calls it `2TM`/`3TM`). `is_primary` is true for exactly one row per player-season:
  the TOT row if it exists, otherwise the only row. ML code filters on `is_primary`. Team
  views can use the per-team rows.
- **Schema changes:** `create_all` never alters an existing table. Until migrations arrive
  (Alembic, level L1), changing a column means `python -m db.init_db --drop` and reloading from
  the HTML cache (`python -m etl.pipeline --force`, ~1.5 min).
- **Lookups are views, not tables.** `teams` and `franchises` (blueprint §7.5) are derived from
  `team_seasons` by `init_db`, so they can never disagree with it. Example:
  `SELECT * FROM teams WHERE franchise_id = 'OKC'` returns `SEA` (1980–2008) and `OKC` (2009–).
  The app's franchise picker reads `franchises`.

---

## Step 4 — Fetching (`etl/scraper.py`)

```python
fetch_page(season, page, refresh=False, sleep_s=4.0) -> str | None
#   GET https://www.basketball-reference.com/leagues/NBA_{season}_{page}.html
#   cache → data/raw/{season}/{page}.html     (a 404 leaves a {page}.404 marker)
```

- **Cache first.** A cached page is never re-downloaded unless `refresh=True`. A full re-parse
  of 47 seasons (`python -m etl.pipeline --force`) makes no requests and takes about 1.5 min,
  versus about 1 h for a real scrape.
- **Politeness.** BRef allows about 20 requests/min and bans offenders for hours. The scraper
  sleeps 4 s after every request (5 s for schedule months), which is about 12 requests/min.
- **Rate limit.** A `429` raises `RateLimited`, and the pipeline stops immediately instead of
  hammering the server.

Wrappers: `get_players` (per-game + advanced pages merged into one frame), `get_standings`
and `get_schedule`. The schedule wrapper first reads the season index
`NBA_{y}_games.html` and discovers the month pages from its links. Fixed month lists break
on lockouts and on the 2020 bubble (`october-2019` … `july`, `august`, `september`,
`october-2020`). The index page already shows the first month, so that month's page is not
fetched again.

About 12 requests per season, about 1 minute per season, so about 1 hour for everything.

---

## Step 5 — Parsing (`etl/parsers.py`)

### 5.1 Read cells by `data-stat`, not by header

Every BRef cell carries a stable `data-stat` key:

```html
<td data-stat="visitor_pts">107</td> … <td data-stat="home_pts">119</td>
```

v1 used `pandas.read_html` and mapped the **first** `PTS` column to the home team. That
column belongs to the visitor, so every result in the database was inverted (§7.1). Reading
`visitor_pts` and `home_pts` by key makes that mistake impossible. `read_row(tr)` turns a
row into `{data-stat: text}` plus three extras: `@href` (link), `@csk` (sort key) and `@id`
(`data-append-csv`, the player id).

**Evidence-driven simplicity:** fallbacks were measured on all 47 cached seasons before the L0
rewrite. Those that never fired were removed: the HTML-comment table search, old-layout
aliases, the conference-table fallback, and backup game-id sources. Resilience against
future layout changes comes back in L2, as data contracts with tests.

### 5.2 Player tables: `parse_player_table(html, "per_game" | "advanced")`

- Table ids: `per_game_stats` and `advanced`.
- Team column: `team_name_abbr` (formerly `Tm`/`team_id`). v1 looked for `Tm` and wrote `UNK`
  for every row (§7.4).
- `TOT`, `2TM`, `3TM`, … → `TOT`. Then `is_primary` and `team_list` (e.g. `"DAL,LAL"`).
- `PER_GAME_STATS` and `ADVANCED_STATS` map our columns to BRef data-stat keys.
- Rows without a player id (the 94 "League Average" rows) are skipped.

### 5.3 Standings: `parse_standings(html)`

- Uses `divs_standings_E/W`, the division tables, which exist in every season since 1980.
  v1 only knew the conference tables, which don't exist before ~2016 (§7.3).
- Each conference is parsed **separately** and only then concatenated. v1 concatenated first,
  and the differing header names wiped out every West row.
- The team abbreviation comes from the link (`/teams/OKC/2025.html`), the division from the
  table's sub-header rows, and `made_playoffs` from the `*` after the name.

### 5.4 Schedule: `parse_schedule(html, season)`

- Skips unplayed games (no score).
- `game_id` comes from the date cell's sort key (`202310240DEN`), and the date is derived
  from it.
- `neutral = is_neutral_site(remarks, season, date)`:
  - **true** for games abroad (BRef remarks such as `at London, England`, `at Mexico City,
    Mexico`) and the 2020 bubble restart (from July 30 2020);
  - **false** for domestic alternate venues (`at Hartford, CT` was effectively a Celtics home)
    and Canadian ones (Raptors and Grizzlies alternates). Their 57.8% home-win rate shows a
    real home edge, unlike games abroad (41.9%) or the bubble (52.3%).

---

## Step 6 — Loading (`etl/loader.py`)

Every load **replaces the season**: in one transaction, `DELETE … WHERE season = ?`, then
insert the fresh rows with `DataFrame.to_sql`.
- **Idempotent:** loading twice gives the same result.
- **Self-cleaning:** rows that vanished from the source vanish from the DB.
- **Atomic:** a crash mid-load rolls back, so a season is never half-loaded.
- v1 used `INSERT … ON CONFLICT DO NOTHING`, which silently kept stale or broken rows.
- Trade-off: replacing ~1,300 games to change one is wasteful. Per-row upserts return in L2
  for in-season updates.

```python
load_players(df, season)     # players upserted by bref_id (they span seasons), then the season's rows
load_standings(df, season)   # adds franchise_id via etl/teams.py
load_games(df, season)       # tags game_type, then writes; raises if the season's standings aren't loaded
```

Player names are always updated (BRef corrects spellings). `birth_year` keeps the earliest
estimate via `LEAST`.

### Game-type inference (`infer_game_types`)

BRef's schedule does not mark playoff games, so the type is inferred per season. It's a pure
function of the date-ordered games and each team's W+L quota (`tests/test_game_types.py`):

1. `remarks` starting with "Play-In" → `play_in`.
2. The season's last `NBA Cup` / `In-Season Tournament` game is the Cup final (2024+). It
   doesn't count in the standings → `cup_final`.
3. Walk the remaining games by date. A game is `regular` while **both** teams are still under
   their W+L total from the standings; everything after → `playoff`.

This is why the pipeline runs **standings before schedule**. Running the schedule step alone
for a season without standings fails loudly, and that step is not checkpointed.

### Franchises (`etl/teams.py`)

`franchise_of("SEA") == "OKC"`. Abbreviations are stored as they were that season, and
`franchise_id` links relocations (NJN→BRK, CHH/CHA→CHO, NOH/NOK→NOP, VAN→MEM, KCK→SAC,
SDC→LAC, WSB→WAS).

---

## Step 7 — Orchestration (`etl/pipeline.py`)

```bash
python -m etl.pipeline                       # every (season, step) not yet done
python -m etl.pipeline --season 2026         # one season
python -m etl.pipeline --start 2000 --end 2010
python -m etl.pipeline --steps schedule      # only some steps
python -m etl.pipeline --season 2024 --force     # ignore the checkpoint, re-parse from cache
python -m etl.pipeline --season 2024 --refresh   # also re-download the pages
```

- **Steps per season:** `players → standings → schedule` (`STEPS = {name: (extract, load)}`).
- **Checkpoint per `(season, step)`, written only on success.** v1 marked a season done even
  when steps failed, which hid the broken standings. A failed step is printed, reported in a
  summary, and retried on the next run. The exit code is 1 if anything failed, or 2 on a rate
  limit.
- **Season range:** `1980 … last_completed_season()`. Season `Y` counts as complete from
  July 1 of `Y`, so a new season is picked up automatically each summer.

---

## Step 8 — Validation gate (`etl/validate.py`)

```bash
python -m etl.validate            # exit code 1 if any check fails
```

**Do not train anything until this passes.** v1 skipped this gate, and inverted scores went
straight into a model.

| Check | Catches |
|---|---|
| All 3 steps checkpointed for every season `1980 … last_completed_season()`; every season has games, standings and players | A season that silently never loaded |
| Home-win rate (regular, non-neutral games) in [0.53, 0.66] overall, ≥ 0.50 per season except 2021 | Inverted scores (v1: 0.394) |
| Every game has a `game_type` and a `neutral` flag; all 2020 bubble games are neutral | Missed inference / migration |
| Standings W/L == W/L recomputed from regular-season games | Cup final or playoff misclassification, missing games |
| 22–30 teams per season across both conferences | Missing West / division tables |
| 1999 = 50 and 2012 = 66 games per team | Lockout handling |
| No `UNK` team, exactly one primary row per player-season (`bref_id` is `NOT NULL` in the schema, so no check needed) | Team column renames, identity bugs |
| 250–700 primary player rows per season; advanced stats wherever minutes were played | Partial loads |
| **Box totals:** Σ(player PPG × games) per team == PS/G × games (±1%; real max error 0.27%) | Missing players, wrong team attribution |
| Traded players: TOT games == sum of their per-team games | Broken `2TM/3TM` handling |
| 30 franchises, all active; every retired abbreviation maps to a franchise | A relocation missing from `etl/teams.py` |
| Spot check: LAL 107 @ DEN 119 on 2023-10-24 | The exact game v1 stored backwards |

Each check is one SQL query in `CHECKS` that returns the rows breaking the rule; it passes when
nothing comes back. Add a row to the `known results` check's `VALUES` list whenever you
verify a result against a box score. `tests/test_validate.py` re-injects past bugs
(swapped scores, missing West, `UNK` teams…) in rolled-back transactions and asserts that
the right check catches each one.

---

## Step 9 — Running it

```bash
python -m db.init_db
python -m etl.pipeline --season 2026     # smoke test one season (~1 min)
python -m etl.validate --seasons 2026
python -m etl.pipeline                   # everything else (~1 h; Ctrl+C-safe, resumes)
python -m etl.validate
pytest -q                                # parsers, game types, scraper, loader (needs TEST_DATABASE_URL)
```

---

## Tests

| File | Needs | Covers |
|---|---|---|
| `test_parsers.py` | nothing | every parser on real pages |
| `test_game_types.py` | nothing | quota logic, play-in, Cup final (both labels), unknown teams |
| `test_scraper.py` | nothing (network faked) | cache hits, `--refresh`, `.404` memory, 429 → `RateLimited` with nothing cached |
| `test_loader.py` | `TEST_DATABASE_URL` (skipped otherwise) | loading twice changes nothing; per-game + advanced fill one wide row; Dončić's TOT/DAL/LAL rows; name corrections propagate; rows dropped from the source are deleted; games need standings; the franchise views |
| `test_validate.py` | a fully loaded `DATABASE_URL` (skipped otherwise) | mutation tests: each re-injected past bug is caught by the right check |

`test_loader.py` drops and recreates the schema in the test DB, and refuses to run if
`TEST_DATABASE_URL` equals `DATABASE_URL`.

The parsers run against real, gzipped BRef pages in `tests/fixtures/`: 1985 and 2025
per-game, 2025 advanced, 1990 and 2025 standings, 2023-24 October and April schedules, and
the 2020 schedule index. They pin every bug above: no `UNK`, the Dončić `TOT` row, both
conferences in 1990, LAL @ DEN not inverted, play-in rows, bubble months, and the
neutral-site rule.

When BRef changes its HTML: save the new page as a fixture, watch the test fail, and fix the
parser. Then `python -m etl.pipeline --force` re-parses the whole cache without touching
the network.

---

## Common Issues

| Problem | Cause | Fix |
|---|---|---|
| `RateLimited: 429` | Too many requests | Wait (BRef bans can last hours), then re-run; the checkpoint resumes. Raise `SLEEP_S` if it recurs |
| `TableNotFound` | BRef renamed a table id or layout | Check the cached page (`grep -o 'table[^>]*id="[^"]*"' data/raw/Y/page.html`) and add the new id or key to the parser |
| `standings W/L ≠ games` in validate | A non-counting game was classified as regular (new tournament format?) | Look at that season's `remarks`, extend `infer_game_types` and add a case to `test_game_types.py` |
| `RuntimeError: no standings for Y` | Schedule step run before standings | `python -m etl.pipeline --season Y` (runs the steps in order) |
| `box totals` fails for one team-season | A player row missing or attributed to the wrong team | Compare that team's rows with the BRef team page; usually a parser key change |
| New column missing in the DB | `create_all` doesn't alter tables | `python -m db.init_db --drop`, then `python -m etl.pipeline --force` (Alembic replaces this in L1) |
| `psycopg2` install fails | Missing libpq | `brew install libpq` |

---

## Verification Queries

```sql
SELECT season, COUNT(*), AVG(home_win::int) FROM games
WHERE game_type = 'regular' AND NOT neutral GROUP BY 1 ORDER BY 1;

SELECT game_type, neutral, COUNT(*) FROM games GROUP BY 1, 2 ORDER BY 1, 2;

SELECT p.name, s.team_list, s.pts, s.per FROM player_seasons s JOIN players p ON p.id = s.player_id
WHERE s.season = 2026 AND s.is_primary ORDER BY s.pts DESC LIMIT 10;

SELECT * FROM franchises WHERE n_abbreviations > 1;          -- relocations

SELECT * FROM etl_checkpoint ORDER BY completed_at DESC LIMIT 8;
```

---

## What Phase 2 Expects From This

A database that passes `etl.validate`: the `games` table with `game_type`, `neutral` and
correct home/away scores, `team_seasons`, and `player_seasons` with `is_primary`. All reads
go through `db.connection.query_df` or the loaders in `ml/features.py`. Next:
`PHASE2_GAME_PREDICTOR_GUIDE.md`.
