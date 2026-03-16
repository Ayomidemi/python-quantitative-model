# Personal Quant Dashboard (NG + US)

Lightweight Python project for your personal portfolio analysis, forecasting, and risk simulation.

## What this does

- Manual input of your current holdings (NG + US) in CSV.
- Pulls prices from Yahoo Finance when possible, with manual fallback.
- Converts positions into one base currency (`NGN` by default).
- Calculates portfolio value, P&L, weights, and annualized volatility.
- Runs quick single-ticker forecasting:
  - 3-month momentum
  - 1-month rolling mean return
  - GARCH(1,1) volatility forecast (if enough data)
- Runs Monte Carlo portfolio simulations.
- Reviews watchlist opportunities since date added.

## Project structure

- `main.py` - CLI entry point.
- `src/data_loader.py` - CSV loading, market price and FX fetching.
- `src/analytics.py` - portfolio metrics and allocation chart.
- `src/forecasting.py` - momentum / mean / GARCH forecast helpers.
- `src/monte_carlo.py` - geometric Brownian motion simulation engine.
- `data/portfolio.csv` - your current holdings.
- `data/watchlist.csv` - your watchlist and missed opportunities.
- `outputs/` - generated plots.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input format

### `data/portfolio.csv`

Required columns:

- `ticker`: your internal label (e.g. AAPL, ZENITHBANK)
- `exchange`: `US` or `NG`
- `shares`: number of shares
- `avg_cost`: your average buy price in local currency
- `currency`: `USD` or `NGN`
- `price_source`: `yfinance` or `manual`
- `symbol_yf`: Yahoo symbol if using yfinance (e.g. AAPL)
- `manual_price`: required if `price_source=manual` or if fetch fails
- `notes`: optional

### `data/watchlist.csv`

Suggested columns:

- `ticker`, `exchange`, `symbol_yf`, `date_added`, `target_price`, `currency`, `notes`

Note: if `symbol_yf` is empty (common for some NG stocks), it will be skipped in automatic return review.

## Commands

### 1) Analyze current portfolio

```bash
python main.py analyze
```

Outputs:

- Portfolio summary in terminal
- Allocation pie chart in `outputs/`

### 2) Forecast one ticker

```bash
python main.py forecast --ticker AAPL --symbol AAPL
```

### 3) Monte Carlo simulation

```bash
python main.py simulate --days 63 --sims 5000 --annual-return 0.12
```

If `--annual-return` is omitted, default is `0.12`.

### 4) Watchlist opportunity review

```bash
python main.py watchlist-review
```

## Personal dashboard frontend (Streamlit)

Run your own local dashboard UI:

```bash
source .venv/bin/activate
streamlit run app/dashboard.py
```

What you get in the UI:

- Portfolio value, P&L, return, annualized volatility
- Positions table and allocation chart
- Single-ticker forecasting panel (momentum, mean return, GARCH/historical vol)
- Monte Carlo simulation controls and risk outcomes

## Notes for NG + US usage

- US symbols generally work directly with Yahoo (`AAPL`, `MSFT`, etc.).
- Some NG symbols may not be available or may use exchange-specific suffixes.
- For NG assets with no reliable API mapping, keep using `price_source=manual`.

## Next upgrades (when ready)

- Add broker CSV import to avoid manual typing.
- Add transaction-level ledger (`buy/sell/dividend`) for realized P&L.
- Add portfolio optimization (min variance, target return).
- Add Streamlit dashboard for UI.