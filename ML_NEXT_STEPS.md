# ML Roadmap

Three tracks, in priority order. Each track is a sequence of feature branches
(`ml/<track>-<step>`), merged into `main` one at a time.

**Ground rules (every step):**
- Never tune on test seasons.
- Always compare with the baselines.
- Log every run in `models/metrics.jsonl` (track X).
- Keep `tests/test_features.py` (the leakage test) green.

---

## Track A — Predict next season's team records *(highest priority)*

**Goal:** before a season starts, predict each team's wins, and beat simple rules first, then
get as close as possible to the betting markets' preseason lines.

**Where we start** (team-seasons 1990–2026; lockout and COVID seasons excluded; wins scaled to
82 games). Mean absolute error in wins:

| Baseline | MAE (wins) |
|---|---|
| Always 41 | 10.48 |
| Same as last season | 8.29 |
| Last season regressed 30% to 41 | **7.66** |
| Linear on last season's net rating | 7.58 (in-sample) |
| **Vegas preseason line** (856 full seasons, 1994–2026) | **6.67**: the real benchmark |

**The gap to close is ~1 win per team (7.66 → 6.67).** Vegas's biggest misses are star
injuries (2025 76ers 50.5 → 24, 1997 Spurs 50.5 → 20), so availability and age risk are where
a model can find an edge. Lines also lean slightly high: unders 50.2% vs overs 45.7%.

**Early finding:** last season's *progression* (wins Y-1 − Y-2) has ~0 correlation (+0.008)
with what the regressed baseline misses. Test it properly, but expect it to be weak.

### A0 — Data *(branch `data/win-totals-payroll`)*
Raw pages are already downloaded into `data/raw/` (see `DATA.md`). Load them:
1. `win_totals` table: season, team, line, over/under odds, actual wins. Source:
   `data/raw/odds/win_totals_{season}.html`, seasons 1994–2026 (no 1999: lockout). The
   archive only fills a season once it ends, so **2026-27 lines need another source**
   (the Kalshi `KXNBAWINS` ladder has them, live; see `SIDE_PROJECTS.md`).
   - Validation: about 30 teams per season, and `actual_wins` equals our standings wins.
   - Map team names → `team_abr` with `team_seasons.full_name`.
2. `team_payrolls` table (team, season, total payroll) from the hidden `salaries2` table on
   `data/raw/{season}/team_{ABR}.html`, 1994–2026. This page type **does** hide its table in an
   HTML comment, so bring back comment search for it, with a test fixture.
3. `salary_caps` table from BRef's `/contracts/salary-cap-history.html` (one page).

### A1 — Baselines and evaluation harness *(branch `ml/season-wins-baselines`)*
`ml/season_wins.py` with a `score(predictions)` function that reports MAE and RMSE in wins
for a season range. Reproduce the table above, **add the Vegas line**, and freeze the split:
train ≤ 2021, validation 2022–2024, test 2025–2026.

### A2 — Features *(branch `ml/season-wins-features`)*, all known before opening night
| Feature | Source | Note |
|---|---|---|
| Last season win%, net rating, SRS, end-of-season Elo | `team_seasons`, game-model snapshots | net rating usually beats record |
| Progression (Y-1 vs Y-2) | `team_seasons` | expect weak (see above) |
| **Roster strength** | `player_seasons` | Σ over the season-Y roster of each player's Y-1 BPM / WS48 / VORP, weighted by his Y-1 minutes |
| Roster continuity | `player_seasons` | share of last season's minutes that returns |
| Age profile | `player_seasons` | minutes-weighted age; young teams improve |
| Projected player stats | career model (track C) | upgrade over raw Y-1 stats |
| Payroll / cap, luxury-tax flag | A0 tables | your "financial flexibility" hypothesis |

⚠️ **Leakage trap (roster):** the season-Y roster built from `player_seasons` knows who
*actually* played for the team, including mid-season trades. Use each player's **first** team
of season Y (the first entry in `team_list`), and weight by **Y-1** minutes, never Y minutes.
Document the remaining leak (a player injured before he ever played is invisible).

### A3 — Model *(branch `ml/season-wins-model`)*
About 950 training rows, so start with **Ridge / linear regression**. Try gradient boosting only
if it beats Ridge on validation. Target: wins per 82 games.
- **Done when:** test MAE clearly beats 7.66, and the gap to Vegas is measured and reported.

### A4 — Product *(branch `feature/season-projections-page`)*
A Streamlit page with 2026-27 projected wins vs the 2026-27 Vegas lines, highlighting the
biggest disagreements.

---

## Track B — Game-by-game model (regular season + playoff mode)

**Now:** XGBoost on Elo/form features, held-out 0.670 accuracy / 0.610 log-loss; Elo alone
0.661 / 0.619.

| Step | Branch | What | Done when |
|---|---|---|---|
| B1 | `ml/elo-tuning` | Tune Elo's K, home advantage, season carry-over and margin multiplier by log-loss on train seasons (grid search). Give the playoffs their own home-advantage value | Elo-only validation log-loss improves; constants documented |
| B2 | `ml/neutral-sites` | Fix the over-reaction to neutral sites: OKC–BOS goes 0.62 → 0.50. Train on each game twice (teams swapped) with a `home_court` = +1/−1/0 input | Equal teams at a neutral site ≈ 0.50; held-out log-loss ≤ 0.610; remove the app warning |
| B3 | `ml/playoff-mode` | Evaluate playoff games separately (n ≈ 80/season). Add series state (game number, series score) and rest; a separate model or an interaction, whichever validates better | Playoff log-loss reported and better than Elo on playoffs |
| B4 | `ml/calibration` | Reliability diagram + calibration error; raw vs isotonic vs sigmoid chosen on one validation season | Choice justified by numbers; calibration error shown in the app |
| B5 | `ml/game-features` | Rest difference, games in the last 7 days, travel | Held-out log-loss < 0.610 |
| B6 | `feature/series-simulator` | Monte-Carlo best-of-7 using playoff mode | Series win % and length distribution in the app |

---

## Track C — Finalize the player models

| Step | Branch | What | Done when |
|---|---|---|---|
| C1 | `ml/projection-vs-ridge` | The career LSTM loses to Ridge (0.364 vs 0.350 normalised MAE): add advanced-stat inputs, use Ridge as the base, light tuning. Ship Ridge if the LSTM still loses | Shipped model beats Ridge on validation, then test. Feeds A2 |
| C2 | `ml/archetype-k` | Choose the number of clusters with silhouette score + stability across seeds | k chosen with a plot and a reason; labels survive a spot-check |
| C3 | `ml/similarity-eval` | Sanity set: 20 known "comps" pairs that should rank high | Retrieval rate reported; regressions caught |

---

## Track X — Experiment log *(do with A1)*
Every `train()` appends one line to `models/metrics.jsonl`: timestamp, git commit, model,
data row counts, validation and test metrics. A notebook plots the history.

## Order
A0 → A1 (+X) → A2 → A3 → B1 → B2 → A4 → B3 → C1 → the rest.
