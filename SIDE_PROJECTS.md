# Side Projects

Two research tracks that reuse this project's data and models. Both share infrastructure with
roadmap Track A (season wins), so do them **after A0–A1**.

---

## 1. How accurate were the preseason betting lines?

**Data:** `data/raw/odds/win_totals_{season}.html`: preseason win totals, over/under odds and
actual wins, 1994–2026 (no 1999 lockout). Loaded into a `win_totals` table in roadmap step A0.

**First results (2026-10-10, 824 team-seasons from full 82-game seasons that have a line):**

| Finding | Number |
|---|---|
| Line MAE | **6.67 wins** (RMSE 8.46). Better than simple rules (7.66), so the market adds ~1 win of information |
| Bias | teams finish **0.41 wins under** their line on average. Unders hit 50.2%, overs 45.7% |
| Stable over time | MAE 6.45 (1994–2003), 6.83 (2004–13), 6.67 (2014–26) |
| Biggest misses | mostly star injuries: 2008 Heat 46.5 → 15, 1997 Spurs 50.5 → 20 (Robinson), 2025 76ers 50.5 → 24 (Embiid); one breakout: 2014 Suns 20.5 → 48 |

**Questions to answer:**
1. **Calibration by line level:** do high lines over-promise (good teams regress) and low lines
   under-promise? Plot actual − line against the line.
2. **Is "always bet under" profitable after the vig?** At −110 you need 52.4% to break even;
   unders hit 50.2% overall. Look for *conditional* edges: high lines, old rosters, teams
   coming off a big jump.
3. **What does the market miss that our data sees?** Regress actual − line on our track A
   features (age, continuity, last season's net rating vs record).
4. **Injury risk:** do lines price the age and availability of a team's top players?

**Done when:** a notebook with the calibration plot, a profitability table after the vig, and
1–2 conditional patterns tested out of sample (train ≤ 2015, check 2016–2026).

---

## 2. Prediction markets: what's there, and where could an edge come from?

### What exists (checked 2026-10-10 via the public APIs)
- **Kalshi** (`api.elections.kalshi.com/trade-api/v2`). Market data needs **no
  authentication**. 364 NBA-related series, including:
  - `KXNBAWINS`, **season win totals** as a ladder per team ("wins ≥ 35, 40, … 70"), already
    live for 2026-27. The ladder gives the market's whole win distribution, which is exactly
    track A's output.
  - `KXNBAGAME` (game winner), `KXNBASPREAD`, `KXNBATOTAL`: track B's territory.
  - `KXNBASERIES` (playoff series winner), `KXNBA` (champion), seeds, awards.
  - Sample (19 of 30 teams; the API pages its results, so use the cursor to fetch the rest):
    about **$944k traded**, **~3¢ spreads** on rungs near 50%, and **no inconsistencies**
    within a ladder. Market medians: SAS 60.5, OKC 60.3, HOU 50.7 … SAC 22.6.
- **Polymarket** (`gamma-api.polymarket.com`). Public read API. Big futures markets (e.g. "NBA
  2027 Champion", ~$24.6M volume) plus player-movement markets.

### Where an edge could come from (most promising first)
1. **Model vs market on season wins.** Turn track A's predicted win distribution into ladder
   probabilities and compare with Kalshi prices. Trade only when the gap is bigger than half
   the spread plus fees. Weak spot: about 30 teams a year, so it takes seasons to tell skill
   from luck.
2. **Cross-market consistency:**
   - Across teams, expected wins must sum to exactly **1,230**. Market-implied wins summing
     above that show a systematic overpricing of "yes".
   - Kalshi vs Polymarket vs sportsbook lines on the same event (champion, win totals).
   - Within a team's ladder (none found today).
3. **Favourite–longshot bias:** cheap longshot contracts (e.g. a 2¢ "champion") are often
   overpriced. Test it on settled markets.
4. **Game markets + injury news:** the market reacts to news faster than any model we can
   build, so the edge there is small. Use them for evaluation (is our game model calibrated
   against market prices?) rather than for trading.
5. **Providing liquidity:** with wide spreads on thin rungs, placing limit orders can beat
   taking prices. It's a different skill (inventory risk), so keep it as a later idea.

### Practical constraints
- Fees and spreads eat small edges, so every comparison must be **after costs**.
- **Capital is locked up**: win-total contracts settle in April, so opportunity cost matters.
- **Eligibility and regulation** differ by platform and country. Check them before trading
  real money. Research and paper trading need no account.
- **History is the bottleneck.** We can't backtest without past prices, so start collecting
  now.

### First steps
1. **`feature/market-snapshots` branch:** a script that saves daily snapshots of
   `KXNBAWINS` (all pages, using the cursor) and `KXNBAGAME` into a `market_prices` table.
   Run it on a schedule (this is roadmap level L2's "scheduled runs" too).
2. Compute the **market-implied median and mean wins** per team from the ladder. This is also
   the 2026-27 line that roadmap A0 couldn't get from the archive.
3. Once track A has a model: **paper-trade** the 2026-27 season by logging hypothetical
   positions and P&L after fees. No real money until a full season of evidence exists.
