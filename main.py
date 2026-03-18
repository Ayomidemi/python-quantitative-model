from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd

from src.analytics import (
    build_portfolio_snapshot,
    estimate_portfolio_volatility,
    portfolio_risk_metrics,
    save_allocation_chart,
)
from src.config import BASE_CURRENCY
from src.data_loader import (
    fetch_fx_rates,
    fetch_prices,
    load_portfolio,
    load_transactions,
    load_watchlist,
)
from src.forecasting import (
    fetch_returns,
    garch_vol_forecast,
    momentum_signal,
    rolling_mean_forecast,
)
from src.ledger import build_ledger_report
from src.monte_carlo import simulate_portfolio_paths
from src.validation import validate_portfolio_inputs, validate_transactions_inputs


def _needed_fx_pairs(currencies: Iterable[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for ccy in currencies:
        c = str(ccy).upper().strip()
        pairs.add((c, BASE_CURRENCY))
    return pairs


def cmd_analyze() -> None:
    portfolio = load_portfolio()
    prices = fetch_prices(portfolio)
    fx = fetch_fx_rates(_needed_fx_pairs(portfolio["currency"].tolist()))
    snapshot = build_portfolio_snapshot(portfolio, prices, fx)

    metrics = portfolio_risk_metrics(snapshot)
    ann_vol = estimate_portfolio_volatility(snapshot, period="1y")
    chart = save_allocation_chart(snapshot)

    print("\n=== Portfolio Summary ===")
    print(f"Base currency: {BASE_CURRENCY}")
    print(f"Total value: {metrics['total_value_base']:,.2f} {BASE_CURRENCY}")
    print(f"Total cost:  {metrics['total_cost_base']:,.2f} {BASE_CURRENCY}")
    print(f"P&L:         {metrics['total_pnl_base']:,.2f} {BASE_CURRENCY}")
    print(f"Return:      {metrics['total_return_pct']:.2f}%")
    if np.isnan(ann_vol):
        print("Annualized vol: N/A (not enough yfinance history)")
    else:
        print(f"Annualized vol: {ann_vol * 100:.2f}%")
    print(f"Allocation chart: {chart}")

    print("\nTop positions:")
    cols = ["ticker", "exchange", "currency", "shares", "market_price", "position_value_base", "weight", "pnl_base"]
    out = snapshot[cols].sort_values("position_value_base", ascending=False).copy()
    out["weight"] = (out["weight"] * 100).round(2)
    out["position_value_base"] = out["position_value_base"].round(2)
    out["pnl_base"] = out["pnl_base"].round(2)
    print(out.to_string(index=False))


def cmd_forecast(ticker: str, symbol: str) -> None:
    returns = fetch_returns(symbol, period="2y")
    if returns.empty:
        raise ValueError(f"No return history found for {symbol}")

    mom_3m = momentum_signal(returns, lookback_days=63)
    mean_1m = rolling_mean_forecast(returns, lookback_days=21)
    garch_vol = garch_vol_forecast(returns, horizon_days=21)
    hist_ann_vol = float(returns.std() * np.sqrt(252.0))

    print(f"\n=== Forecast for {ticker} ({symbol}) ===")
    print(f"3M momentum: {mom_3m * 100:.2f}%" if mom_3m is not None else "3M momentum: N/A")
    print(f"1M mean return forecast (daily): {mean_1m * 100:.3f}%" if mean_1m is not None else "1M mean return forecast: N/A")
    if garch_vol is None:
        print(f"GARCH annualized vol forecast: N/A (fallback to historical {hist_ann_vol * 100:.2f}%)")
    else:
        print(f"GARCH annualized vol forecast: {garch_vol * 100:.2f}%")


def cmd_simulate(days: int, sims: int, annual_return: float | None) -> None:
    portfolio = load_portfolio()
    prices = fetch_prices(portfolio)
    fx = fetch_fx_rates(_needed_fx_pairs(portfolio["currency"].tolist()))
    snapshot = build_portfolio_snapshot(portfolio, prices, fx)
    metrics = portfolio_risk_metrics(snapshot)
    ann_vol = estimate_portfolio_volatility(snapshot, period="1y")

    if np.isnan(ann_vol):
        ann_vol = 0.25
    ann_ret = annual_return if annual_return is not None else 0.12
    result = simulate_portfolio_paths(
        initial_value=metrics["total_value_base"],
        annual_return=ann_ret,
        annual_volatility=ann_vol,
        horizon_days=days,
        num_sims=sims,
    )

    print(f"\n=== Monte Carlo ({days} trading days, {sims} sims) ===")
    print(f"Initial value:       {metrics['total_value_base']:,.2f} {BASE_CURRENCY}")
    print(f"Expected final:      {result.expected_final_value:,.2f} {BASE_CURRENCY}")
    print(f"P5 / Median / P95:   {result.p5_final_value:,.2f} / {result.p50_final_value:,.2f} / {result.p95_final_value:,.2f}")
    print(f"Probability of loss: {result.prob_loss * 100:.2f}%")
    print(f"Assumed annual return: {ann_ret * 100:.2f}%")
    print(f"Assumed annual vol:    {ann_vol * 100:.2f}%")


def cmd_watchlist_review() -> None:
    wl = load_watchlist()
    if "symbol_yf" not in wl.columns:
        raise ValueError("watchlist.csv must include symbol_yf column.")
    if "date_added" not in wl.columns:
        raise ValueError("watchlist.csv must include date_added column.")

    rows = []
    for _, row in wl.iterrows():
        raw_symbol = row["symbol_yf"]
        if pd.isna(raw_symbol):
            continue
        symbol = str(raw_symbol).strip()
        if not symbol or symbol.lower() == "nan":
            continue
        ticker = str(row["ticker"]).strip()
        date_added = pd.to_datetime(row["date_added"], errors="coerce")
        if pd.isna(date_added) or not symbol:
            continue

        hist = fetch_returns(symbol, period="2y")
        if hist.empty:
            continue
        price_hist = (1 + hist).cumprod()
        price_hist.index = pd.to_datetime(price_hist.index).tz_localize(None)
        mask = price_hist.index >= date_added
        if not mask.any():
            continue
        seg = price_hist.loc[mask]
        move = float(seg.iloc[-1] / seg.iloc[0] - 1.0)
        rows.append({"ticker": ticker, "symbol": symbol, "return_since_add_pct": move * 100.0})

    if not rows:
        print("No watchlist opportunities could be evaluated yet.")
        return

    review = pd.DataFrame(rows).sort_values("return_since_add_pct", ascending=False)
    print("\n=== Watchlist Opportunity Review ===")
    print(review.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


def cmd_ledger_summary() -> None:
    portfolio = load_portfolio()
    transactions = load_transactions()
    prices = fetch_prices(portfolio)
    all_ccy = set(portfolio["currency"].tolist()) | set(transactions["currency"].tolist())
    fx = fetch_fx_rates(_needed_fx_pairs(all_ccy))
    snapshot = build_portfolio_snapshot(portfolio, prices, fx)

    report = build_ledger_report(transactions, snapshot, fx)
    totals = report.totals
    print("\n=== Ledger P&L Summary ===")
    print(f"Base currency: {BASE_CURRENCY}")
    print(f"Realized P&L:   {totals['realized_pnl_base']:,.2f} {BASE_CURRENCY}")
    print(f"Unrealized P&L: {totals['unrealized_pnl_base']:,.2f} {BASE_CURRENCY}")
    print(f"Total P&L:      {totals['total_pnl_base']:,.2f} {BASE_CURRENCY}")
    print(f"Market Value:   {totals['market_value_base']:,.2f} {BASE_CURRENCY}")

    if report.positions.empty:
        print("\nNo transactions yet. Add rows to data/transactions.csv.")
        return

    cols = [
        "ticker",
        "open_quantity",
        "avg_cost_local",
        "market_price",
        "realized_pnl_base",
        "unrealized_pnl_base",
        "total_pnl_base",
    ]
    out = report.positions[cols].copy()
    print("\nBy ticker:")
    print(out.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


def cmd_validate_data() -> None:
    portfolio = load_portfolio()
    transactions = load_transactions()

    p_report = validate_portfolio_inputs(portfolio)
    t_report = validate_transactions_inputs(transactions)

    print("\n=== Data Validation Report ===")

    if p_report.errors:
        print("\nPortfolio errors:")
        for msg in p_report.errors:
            print(f"- {msg}")
    if p_report.warnings:
        print("\nPortfolio warnings:")
        for msg in p_report.warnings:
            print(f"- {msg}")

    if t_report.errors:
        print("\nTransaction errors:")
        for msg in t_report.errors:
            print(f"- {msg}")
    if t_report.warnings:
        print("\nTransaction warnings:")
        for msg in t_report.warnings:
            print(f"- {msg}")

    if not (p_report.errors or p_report.warnings or t_report.errors or t_report.warnings):
        print("No issues found. Inputs look clean.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Personal Quant Dashboard")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("analyze", help="Analyze current portfolio from CSV inputs")

    forecast = sub.add_parser("forecast", help="Run quick single-ticker forecast")
    forecast.add_argument("--ticker", required=True, help="Friendly ticker name")
    forecast.add_argument("--symbol", required=True, help="Yahoo Finance symbol")

    sim = sub.add_parser("simulate", help="Run Monte Carlo on current portfolio")
    sim.add_argument("--days", type=int, default=63, help="Horizon in trading days")
    sim.add_argument("--sims", type=int, default=5000, help="Number of simulation paths")
    sim.add_argument(
        "--annual-return",
        type=float,
        default=None,
        help="Optional annual expected return (decimal, e.g. 0.12)",
    )

    sub.add_parser("watchlist-review", help="Evaluate watchlist misses since add date")
    sub.add_parser("ledger-summary", help="Summarize realized and unrealized P&L from transactions")
    sub.add_parser("validate-data", help="Run lightweight data health checks before analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "analyze":
        cmd_analyze()
    elif args.command == "forecast":
        cmd_forecast(args.ticker, args.symbol)
    elif args.command == "simulate":
        cmd_simulate(args.days, args.sims, args.annual_return)
    elif args.command == "watchlist-review":
        cmd_watchlist_review()
    elif args.command == "ledger-summary":
        cmd_ledger_summary()
    elif args.command == "validate-data":
        cmd_validate_data()


if __name__ == "__main__":
    main()
