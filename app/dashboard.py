from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure `src` imports work when Streamlit executes from `app/`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analytics import (
    build_portfolio_snapshot,
    estimate_portfolio_volatility,
    portfolio_risk_metrics,
)
from src.config import BASE_CURRENCY
from src.data_loader import fetch_fx_rates, fetch_prices, load_portfolio, load_watchlist
from src.forecasting import (
    fetch_returns,
    garch_vol_forecast,
    momentum_signal,
    rolling_mean_forecast,
)
from src.monte_carlo import simulate_portfolio_paths


st.set_page_config(page_title="Personal Quant Dashboard", page_icon="📈", layout="wide")


def needed_fx_pairs(currencies: list[str]) -> set[tuple[str, str]]:
    return {(str(ccy).upper().strip(), BASE_CURRENCY) for ccy in currencies}


@st.cache_data(ttl=300)
def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    portfolio = load_portfolio()
    watchlist = load_watchlist()
    prices = fetch_prices(portfolio)
    fx = fetch_fx_rates(needed_fx_pairs(portfolio["currency"].tolist()))
    snapshot = build_portfolio_snapshot(portfolio, prices, fx)
    return portfolio, watchlist, snapshot


def render_header(snapshot: pd.DataFrame) -> None:
    metrics = portfolio_risk_metrics(snapshot)
    ann_vol = estimate_portfolio_volatility(snapshot, period="1y")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Value", f"{metrics['total_value_base']:,.2f} {BASE_CURRENCY}")
    c2.metric("Total P&L", f"{metrics['total_pnl_base']:,.2f} {BASE_CURRENCY}")
    c3.metric("Return", f"{metrics['total_return_pct']:.2f}%")
    c4.metric(
        "Annualized Vol",
        "N/A" if np.isnan(ann_vol) else f"{ann_vol * 100:.2f}%",
    )


def render_portfolio_table(snapshot: pd.DataFrame) -> None:
    view = snapshot[
        [
            "ticker",
            "exchange",
            "currency",
            "shares",
            "market_price",
            "position_value_base",
            "weight",
            "pnl_base",
        ]
    ].copy()
    view["weight"] = view["weight"] * 100.0
    view = view.sort_values("position_value_base", ascending=False)

    st.subheader("Positions")
    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "market_price": st.column_config.NumberColumn("Market Price", format="%.2f"),
            "position_value_base": st.column_config.NumberColumn(
                f"Value ({BASE_CURRENCY})", format="%.2f"
            ),
            "weight": st.column_config.NumberColumn("Weight %", format="%.2f"),
            "pnl_base": st.column_config.NumberColumn(
                f"P&L ({BASE_CURRENCY})", format="%.2f"
            ),
        },
    )


def render_allocation(snapshot: pd.DataFrame) -> None:
    alloc = (
        snapshot.groupby("ticker", as_index=False)["position_value_base"]
        .sum()
        .sort_values("position_value_base", ascending=False)
    )
    st.subheader("Allocation")
    st.bar_chart(alloc, x="ticker", y="position_value_base")


def render_forecast(portfolio: pd.DataFrame, watchlist: pd.DataFrame) -> None:
    st.subheader("Single-Ticker Forecast")
    symbols = {}
    if "symbol_yf" in portfolio.columns:
        for _, row in portfolio.iterrows():
            symbol = row.get("symbol_yf")
            if pd.notna(symbol):
                s = str(symbol).strip()
                if s and s.lower() != "nan":
                    symbols[f"{row['ticker']} ({s})"] = s
    if "symbol_yf" in watchlist.columns:
        for _, row in watchlist.iterrows():
            symbol = row.get("symbol_yf")
            if pd.notna(symbol):
                s = str(symbol).strip()
                if s and s.lower() != "nan":
                    symbols[f"{row['ticker']} ({s}) [watchlist]"] = s

    if not symbols:
        st.info("No valid Yahoo symbols found in portfolio/watchlist.")
        return

    pick = st.selectbox("Choose symbol", options=list(symbols.keys()))
    symbol = symbols[pick]
    returns = fetch_returns(symbol, period="2y")
    if returns.empty:
        st.warning("Could not fetch return history for selected symbol.")
        return

    mom_3m = momentum_signal(returns, lookback_days=63)
    mean_1m = rolling_mean_forecast(returns, lookback_days=21)
    garch_vol = garch_vol_forecast(returns, horizon_days=21)
    hist_vol = float(returns.std() * np.sqrt(252.0))

    c1, c2, c3 = st.columns(3)
    c1.metric("3M Momentum", "N/A" if mom_3m is None else f"{mom_3m * 100:.2f}%")
    c2.metric(
        "1M Mean Daily Return",
        "N/A" if mean_1m is None else f"{mean_1m * 100:.3f}%",
    )
    c3.metric(
        "Annualized Vol",
        f"{hist_vol * 100:.2f}%" if garch_vol is None else f"{garch_vol * 100:.2f}% (GARCH)",
    )

    cumulative = (1.0 + returns).cumprod()
    st.line_chart(cumulative.rename("Cumulative growth"))


def render_monte_carlo(snapshot: pd.DataFrame) -> None:
    st.subheader("Monte Carlo")
    ann_vol = estimate_portfolio_volatility(snapshot, period="1y")
    if np.isnan(ann_vol):
        ann_vol = 0.25

    left, right = st.columns(2)
    days = left.slider("Horizon (trading days)", min_value=21, max_value=252, value=63, step=21)
    sims = right.slider("Number of simulations", min_value=1000, max_value=20000, value=5000, step=1000)
    ann_return = st.slider("Assumed annual return", min_value=-0.20, max_value=0.60, value=0.12, step=0.01)

    metrics = portfolio_risk_metrics(snapshot)
    result = simulate_portfolio_paths(
        initial_value=metrics["total_value_base"],
        annual_return=ann_return,
        annual_volatility=ann_vol,
        horizon_days=days,
        num_sims=sims,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Final", f"{result.expected_final_value:,.2f} {BASE_CURRENCY}")
    c2.metric("P5", f"{result.p5_final_value:,.2f} {BASE_CURRENCY}")
    c3.metric("Median", f"{result.p50_final_value:,.2f} {BASE_CURRENCY}")
    c4.metric("Loss Probability", f"{result.prob_loss * 100:.2f}%")


def main() -> None:
    st.title("📈 Personal Quant Dashboard")
    st.caption("Portfolio analytics, forecasting, and simulation for your personal use.")

    try:
        portfolio, watchlist, snapshot = load_all_data()
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

    render_header(snapshot)
    st.divider()
    render_portfolio_table(snapshot)
    st.divider()
    render_allocation(snapshot)
    st.divider()
    render_forecast(portfolio, watchlist)
    st.divider()
    render_monte_carlo(snapshot)

    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
