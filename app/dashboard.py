from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
from src.config import BASE_CURRENCY, DATA_DIR
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
from src.monte_carlo import simulate_portfolio_paths
from src.ledger import TRANSACTION_COLUMNS, build_ledger_report, normalize_transactions


st.set_page_config(page_title="Personal Quant Dashboard", page_icon="📈", layout="wide")

PORTFOLIO_PATH = DATA_DIR / "portfolio.csv"
TRANSACTIONS_PATH = DATA_DIR / "transactions.csv"
PORTFOLIO_COLUMNS = [
    "ticker",
    "exchange",
    "shares",
    "avg_cost",
    "currency",
    "price_source",
    "symbol_yf",
    "manual_price",
    "notes",
]


def needed_fx_pairs(currencies: list[str]) -> set[tuple[str, str]]:
    return {(str(ccy).upper().strip(), BASE_CURRENCY) for ccy in currencies}


@st.cache_data(ttl=300)
def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str], float]]:
    portfolio = load_portfolio()
    watchlist = load_watchlist()
    transactions = load_transactions()
    prices = fetch_prices(portfolio)
    all_currencies = set(portfolio["currency"].tolist()) | set(transactions["currency"].tolist())
    fx = fetch_fx_rates(needed_fx_pairs(list(all_currencies)))
    snapshot = build_portfolio_snapshot(portfolio, prices, fx)
    return portfolio, watchlist, transactions, snapshot, fx


def render_header(snapshot: pd.DataFrame) -> None:
    metrics = portfolio_risk_metrics(snapshot)
    ann_vol = estimate_portfolio_volatility(snapshot, period="1y")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Value", f"{metrics['total_value_base']:,.2f} {BASE_CURRENCY}")
    c2.metric("Total P&L", f"{metrics['total_pnl_base']:,.2f} {BASE_CURRENCY}")
    c3.metric(
        "Return",
        "N/A (set avg_cost first)"
        if np.isnan(metrics["total_return_pct"])
        else f"{metrics['total_return_pct']:.2f}%",
    )
    c4.metric(
        "Annualized Vol",
        "N/A" if np.isnan(ann_vol) else f"{ann_vol * 100:.2f}%",
    )


def _normalize_portfolio_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in PORTFOLIO_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[PORTFOLIO_COLUMNS]

    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["exchange"] = out["exchange"].astype(str).str.strip().str.upper()
    out["currency"] = out["currency"].astype(str).str.strip().str.upper()
    out["price_source"] = out["price_source"].astype(str).str.strip().str.lower()
    out["symbol_yf"] = out["symbol_yf"].fillna("").astype(str).str.strip()
    out["notes"] = out["notes"].fillna("").astype(str).str.strip()

    out["shares"] = pd.to_numeric(out["shares"], errors="coerce")
    out["avg_cost"] = pd.to_numeric(out["avg_cost"], errors="coerce")
    out["manual_price"] = pd.to_numeric(out["manual_price"], errors="coerce")

    out = out[out["ticker"] != ""]
    out = out.dropna(subset=["shares", "avg_cost"])
    out = out[out["price_source"].isin(["manual", "yfinance"])]

    invalid = out[(out["price_source"] == "yfinance") & (out["symbol_yf"].str.strip() == "")]
    if not invalid.empty:
        raise ValueError("Rows with price_source=yfinance must include symbol_yf.")
    return out


def _save_portfolio(df: pd.DataFrame) -> None:
    clean = _normalize_portfolio_df(df)
    if clean.empty:
        raise ValueError("Portfolio cannot be empty after validation.")
    clean.to_csv(PORTFOLIO_PATH, index=False)


def _normalize_transactions_df(df: pd.DataFrame) -> pd.DataFrame:
    clean = normalize_transactions(df)
    # Keep expected columns even if empty after filtering
    for col in TRANSACTION_COLUMNS:
        if col not in clean.columns:
            clean[col] = np.nan
    return clean[TRANSACTION_COLUMNS]


def _save_transactions(df: pd.DataFrame) -> None:
    clean = _normalize_transactions_df(df)
    clean.to_csv(TRANSACTIONS_PATH, index=False)


@st.dialog("Add Portfolio Position")
def add_position_dialog(existing_df: pd.DataFrame) -> None:
    with st.form("add_position_form", clear_on_submit=True):
        st.caption("Add a new ticker row to your portfolio inputs.")
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Ticker", placeholder="AAPL")
        exchange = c2.selectbox("Exchange", options=["US", "NG"])
        currency = c3.selectbox("Currency", options=["USD", "NGN"])

        c4, c5, c6 = st.columns(3)
        shares = c4.number_input("Shares", min_value=0.0, value=0.0, step=1.0)
        avg_cost = c5.number_input("Average Cost", min_value=0.0, value=0.0, step=0.01)
        price_source = c6.selectbox("Price Source", options=["yfinance", "manual"])

        c7, c8 = st.columns(2)
        symbol_yf = c7.text_input("Yahoo Symbol (if yfinance)", placeholder="AAPL")
        manual_price = c8.number_input("Manual Price (optional)", min_value=0.0, value=0.0, step=0.01)
        notes = st.text_input("Notes (optional)")

        submitted = st.form_submit_button("Save Position", type="primary")
        if submitted:
            try:
                row: dict[str, Any] = {
                    "ticker": ticker,
                    "exchange": exchange,
                    "shares": shares,
                    "avg_cost": avg_cost,
                    "currency": currency,
                    "price_source": price_source,
                    "symbol_yf": symbol_yf if price_source == "yfinance" else "",
                    "manual_price": manual_price if manual_price > 0 else np.nan,
                    "notes": notes,
                }
                updated = pd.concat([existing_df, pd.DataFrame([row])], ignore_index=True)
                _save_portfolio(updated)
                st.cache_data.clear()
                st.success("Position saved. Recalculating...")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save new row: {exc}")


def render_portfolio_inputs(portfolio: pd.DataFrame) -> None:
    st.subheader("Portfolio Inputs")
    st.caption("Edit rows directly, or add a row with the modal. Save to recalculate everything.")
    st.info(
        "Tip: set `avg_cost` for each position to unlock accurate P&L/return. "
        "If a Nigerian ticker does not load from Yahoo, switch `price_source` to `manual` and set `manual_price`."
    )

    b1, b2 = st.columns([1, 1])
    if b1.button("Add Position", use_container_width=True):
        add_position_dialog(portfolio.copy())

    editable = portfolio.copy()
    for col in PORTFOLIO_COLUMNS:
        if col not in editable.columns:
            editable[col] = np.nan
    editable = editable[PORTFOLIO_COLUMNS]

    edited = st.data_editor(
        editable,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "exchange": st.column_config.SelectboxColumn("Exchange", options=["US", "NG"]),
            "currency": st.column_config.SelectboxColumn("Currency", options=["USD", "NGN"]),
            "price_source": st.column_config.SelectboxColumn(
                "Price Source", options=["yfinance", "manual"]
            ),
            "shares": st.column_config.NumberColumn("Shares", min_value=0.0, format="%.4f"),
            "avg_cost": st.column_config.NumberColumn("Avg Cost", min_value=0.0, format="%.4f"),
            "manual_price": st.column_config.NumberColumn(
                "Manual Price", min_value=0.0, format="%.4f"
            ),
        },
    )

    if b2.button("Save Inputs", type="primary", use_container_width=True):
        try:
            _save_portfolio(pd.DataFrame(edited))
            st.cache_data.clear()
            st.success("Portfolio inputs saved.")
            st.rerun()
        except Exception as exc:
            st.error(f"Save failed: {exc}")


@st.dialog("Add Transaction")
def add_transaction_dialog(existing_df: pd.DataFrame) -> None:
    with st.form("add_transaction_form", clear_on_submit=True):
        st.caption("Record a buy or sell to compute realized/unrealized P&L.")
        c1, c2, c3 = st.columns(3)
        date = c1.date_input("Date")
        ticker = c2.text_input("Ticker", placeholder="AMZN")
        exchange = c3.selectbox("Exchange", options=["US", "NG"])

        c4, c5, c6 = st.columns(3)
        side = c4.selectbox("Side", options=["buy", "sell"])
        quantity = c5.number_input("Quantity", min_value=0.0, value=0.0, step=0.0001, format="%.8f")
        price = c6.number_input("Price", min_value=0.0, value=0.0, step=0.01)

        c7, c8 = st.columns(2)
        currency = c7.selectbox("Currency", options=["USD", "NGN"])
        fee = c8.number_input("Fee", min_value=0.0, value=0.0, step=0.01)
        notes = st.text_input("Notes (optional)")

        submitted = st.form_submit_button("Save Transaction", type="primary")
        if submitted:
            row: dict[str, Any] = {
                "date": date,
                "ticker": ticker,
                "exchange": exchange,
                "side": side,
                "quantity": quantity,
                "price": price,
                "currency": currency,
                "fee": fee,
                "notes": notes,
            }
            try:
                updated = pd.concat([existing_df, pd.DataFrame([row])], ignore_index=True)
                _save_transactions(updated)
                st.cache_data.clear()
                st.success("Transaction saved.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save transaction: {exc}")


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


@st.dialog("Set Missing NG Prices")
def set_missing_ng_prices_dialog(pending_ng: pd.DataFrame, portfolio: pd.DataFrame) -> None:
    st.caption("Enter latest market prices in NGN. These rows will be switched to manual pricing.")
    defaults: dict[str, float] = {}
    for _, row in pending_ng.iterrows():
        ticker = str(row["ticker"])
        current = portfolio.loc[portfolio["ticker"] == ticker, "manual_price"]
        base = 0.0
        if not current.empty and pd.notna(current.iloc[0]):
            try:
                base = float(current.iloc[0])
            except Exception:
                base = 0.0
        defaults[ticker] = base

    with st.form("set_missing_ng_prices_form"):
        price_inputs: dict[str, float] = {}
        for ticker, default in defaults.items():
            price_inputs[ticker] = st.number_input(
                f"{ticker} price (NGN)",
                min_value=0.0,
                value=float(default),
                step=0.01,
                key=f"missing_ng_{ticker}",
            )

        submitted = st.form_submit_button("Save NG Prices", type="primary")
        if submitted:
            invalid = [t for t, p in price_inputs.items() if p <= 0]
            if invalid:
                st.error(f"Please enter prices > 0 for: {', '.join(invalid)}")
                return
            updated = portfolio.copy()
            for ticker, price in price_inputs.items():
                mask = (updated["ticker"].astype(str).str.upper() == ticker.upper()) & (
                    updated["exchange"].astype(str).str.upper() == "NG"
                )
                updated.loc[mask, "price_source"] = "manual"
                updated.loc[mask, "manual_price"] = float(price)

            try:
                _save_portfolio(updated)
                st.cache_data.clear()
                st.success("NG prices saved. Recalculating...")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save NG prices: {exc}")


def render_data_quality(snapshot: pd.DataFrame, portfolio: pd.DataFrame) -> None:
    pending = snapshot[snapshot["market_price"] <= 0].copy()
    if pending.empty:
        st.success("All positions have non-zero prices.")
        return

    st.warning(
        "Some positions are priced at 0.00. Set `manual_price` (and switch to `manual` source if needed) "
        "to include them in valuation."
    )
    st.dataframe(
        pending[["ticker", "exchange", "price_source", "symbol_yf", "manual_price", "notes"]],
        use_container_width=True,
        hide_index=True,
    )
    pending_ng = pending[pending["exchange"].astype(str).str.upper() == "NG"]
    if not pending_ng.empty and st.button("Set Missing NG Prices", type="primary"):
        set_missing_ng_prices_dialog(pending_ng, portfolio)


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


def render_ledger(transactions: pd.DataFrame, snapshot: pd.DataFrame, fx: dict[tuple[str, str], float]) -> None:
    st.subheader("Transaction Ledger")
    st.caption("Use transactions for true realized/unrealized P&L instead of only static holdings.")
    st.info(
        "Rule: add every buy/sell here. Realized P&L is calculated from sells against your running cost basis."
    )

    b1, b2 = st.columns([1, 1])
    if b1.button("Add Transaction", use_container_width=True):
        add_transaction_dialog(transactions.copy())

    editable = transactions.copy()
    for col in TRANSACTION_COLUMNS:
        if col not in editable.columns:
            editable[col] = np.nan
    editable = editable[TRANSACTION_COLUMNS]

    edited = st.data_editor(
        editable,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "side": st.column_config.SelectboxColumn("Side", options=["buy", "sell"]),
            "exchange": st.column_config.SelectboxColumn("Exchange", options=["US", "NG"]),
            "currency": st.column_config.SelectboxColumn("Currency", options=["USD", "NGN"]),
            "quantity": st.column_config.NumberColumn("Quantity", min_value=0.0, format="%.8f"),
            "price": st.column_config.NumberColumn("Price", min_value=0.0, format="%.4f"),
            "fee": st.column_config.NumberColumn("Fee", min_value=0.0, format="%.4f"),
        },
    )

    if b2.button("Save Transactions", type="primary", use_container_width=True):
        try:
            _save_transactions(pd.DataFrame(edited))
            st.cache_data.clear()
            st.success("Transactions saved.")
            st.rerun()
        except Exception as exc:
            st.error(f"Save failed: {exc}")

    try:
        report = build_ledger_report(transactions, snapshot, fx)
    except Exception as exc:
        st.error(f"Ledger calculation error: {exc}")
        return

    totals = report.totals
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Realized P&L", f"{totals['realized_pnl_base']:,.2f} {BASE_CURRENCY}")
    c2.metric("Unrealized P&L", f"{totals['unrealized_pnl_base']:,.2f} {BASE_CURRENCY}")
    c3.metric("Total P&L", f"{totals['total_pnl_base']:,.2f} {BASE_CURRENCY}")
    c4.metric("Open Market Value", f"{totals['market_value_base']:,.2f} {BASE_CURRENCY}")

    if report.positions.empty:
        st.warning("No valid transactions yet. Add buys/sells to unlock true P&L tracking.")
        return

    view = report.positions[
        [
            "ticker",
            "open_quantity",
            "avg_cost_local",
            "market_price",
            "realized_pnl_base",
            "unrealized_pnl_base",
            "total_pnl_base",
        ]
    ].copy()
    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "open_quantity": st.column_config.NumberColumn("Open Qty", format="%.8f"),
            "avg_cost_local": st.column_config.NumberColumn("Avg Cost", format="%.4f"),
            "market_price": st.column_config.NumberColumn("Market Price", format="%.4f"),
            "realized_pnl_base": st.column_config.NumberColumn("Realized P&L", format="%.2f"),
            "unrealized_pnl_base": st.column_config.NumberColumn("Unrealized P&L", format="%.2f"),
            "total_pnl_base": st.column_config.NumberColumn("Total P&L", format="%.2f"),
        },
    )


def main() -> None:
    st.title("📈 Personal Quant Dashboard")
    st.caption("Portfolio analytics, forecasting, and simulation for your personal use.")

    with st.sidebar:
        st.header("Workflow")
        st.markdown(
            "1. Update holdings in **Portfolio Inputs**\n"
            "2. Click **Save Inputs**\n"
            "3. Review **Overview**, **Forecast**, and **Simulation**"
        )
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    try:
        portfolio, watchlist, transactions, snapshot, fx = load_all_data()
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

    tab_overview, tab_inputs, tab_ledger, tab_forecast, tab_sim, tab_watch = st.tabs(
        ["Overview", "Portfolio Inputs", "Ledger", "Forecast", "Simulation", "Watchlist"]
    )

    with tab_overview:
        render_header(snapshot)
        st.divider()
        render_data_quality(snapshot, portfolio)
        st.divider()
        render_portfolio_table(snapshot)
        st.divider()
        render_allocation(snapshot)

    with tab_inputs:
        render_portfolio_inputs(portfolio)

    with tab_ledger:
        render_ledger(transactions, snapshot, fx)

    with tab_forecast:
        render_forecast(portfolio, watchlist)

    with tab_sim:
        render_monte_carlo(snapshot)

    with tab_watch:
        st.subheader("Watchlist Data")
        st.dataframe(watchlist, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
