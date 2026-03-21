# Value sleeve — simple playbook

Use with `python main.py value-screen` (ranks watchlist names by **lower P/E + lower P/B** vs each other).

## What “value” means here

- **Cheap vs history or peers** (P/E, P/B) — not automatically “good.”
- **Traps:** distressed banks, dying industries, accounting quirks → low ratios forever.
- **Your edge:** policy + position size + **sell rules** + (optional) reading filings/news.

## What to buy (process)

1. Run **`value-screen`** on a watchlist of names you already understand (sector, business).
2. Keep only names that pass **quality filters** you define (e.g. positive earnings, manageable debt — often manual at first).
3. **Size small**; respect `policy.md` max weight.
4. Prefer **liquid** US names/ETFs so spreads + Bamboo fees hurt less.

## When to sell (rules beat hope)

Define at buy time:

| Rule | Example |
|------|--------|
| **Thesis broken** | Earnings miss + guidance down + you no longer trust the story → exit |
| **Valuation** | P/E above your sell band or back to “fair” from your model |
| **Time** | Review quarterly; if still cheap but no progress after N quarters, reassess |
| **Stop** | Hard max loss % on the position (value can stay “cheap” and fall more) |

## What not to do

- Buy **only** because ratio is low (value trap).
- Average down **without** a new thesis (write it down first).
- Ignore **fees** — many small trades destroy value on small accounts.

## Repo tools

- `value-screen` — quick rank from Yahoo fundamentals (data can be missing/wrong; verify).
- `validate-data` — before you act on portfolio inputs.
- Ledger — log every buy/sell so realized P&L is real.
