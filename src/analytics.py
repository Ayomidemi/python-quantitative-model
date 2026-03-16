from __future__ import annotations

import io
from datetime import datetime
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import BASE_CURRENCY, OUTPUT_DIR


def build_portfolio_snapshot(
    portfolio_df: pd.DataFrame,
    prices: Dict[str, float],
    fx_rates: Dict[tuple[str, str], float],
) -> pd.DataFrame:
    df = portfolio_df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["market_price"] = df["ticker"].map(prices)
    if df["market_price"].isna().any():
        missing = df.loc[df["market_price"].isna(), "ticker"].tolist()
        raise ValueError(f"Missing market prices for tickers: {missing}")

    def to_base_currency(row: pd.Series) -> float:
        ccy = str(row["currency"]).upper().strip()
        rate = fx_rates.get((ccy, BASE_CURRENCY), 1.0 if ccy == BASE_CURRENCY else None)
        if rate is None:
            raise ValueError(f"No FX conversion from {ccy} to {BASE_CURRENCY}")
        return float(rate)

    df["fx_to_base"] = df.apply(to_base_currency, axis=1)
    df["position_value_local"] = df["shares"] * df["market_price"]
    df["position_value_base"] = df["position_value_local"] * df["fx_to_base"]
    df["cost_value_local"] = df["shares"] * df["avg_cost"]
    df["cost_value_base"] = df["cost_value_local"] * df["fx_to_base"]
    df["pnl_base"] = df["position_value_base"] - df["cost_value_base"]
    total = float(df["position_value_base"].sum())
    df["weight"] = df["position_value_base"] / total if total > 0 else 0.0
    return df


def _download_returns(symbols: Dict[str, str], period: str = "1y") -> pd.DataFrame:
    series_map: Dict[str, pd.Series] = {}
    for ticker, symbol in symbols.items():
        if not symbol or pd.isna(symbol):
            continue
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            hist = yf.Ticker(symbol).history(period=period)
        if hist.empty:
            continue
        ret = hist["Close"].pct_change().dropna()
        if ret.empty:
            continue
        series_map[ticker] = ret
    if not series_map:
        return pd.DataFrame()
    return pd.concat(series_map, axis=1).dropna(how="all")


def portfolio_risk_metrics(snapshot_df: pd.DataFrame) -> Dict[str, float]:
    total_value = float(snapshot_df["position_value_base"].sum())
    total_cost = float(snapshot_df["cost_value_base"].sum())
    total_pnl = float(snapshot_df["pnl_base"].sum())
    ret = (total_value / total_cost - 1.0) if total_cost > 0 else np.nan
    return {
        "total_value_base": total_value,
        "total_cost_base": total_cost,
        "total_pnl_base": total_pnl,
        "total_return_pct": ret * 100.0 if not np.isnan(ret) else np.nan,
    }


def estimate_portfolio_volatility(snapshot_df: pd.DataFrame, period: str = "1y") -> float:
    symbols = {
        str(row["ticker"]).strip(): str(row["symbol_yf"]).strip()
        for _, row in snapshot_df.iterrows()
        if str(row["price_source"]).lower().strip() == "yfinance"
    }
    rets = _download_returns(symbols, period=period)
    if rets.empty:
        return np.nan

    w = snapshot_df.set_index("ticker")["weight"].reindex(rets.columns).fillna(0.0).values
    cov_daily = rets.cov().values
    vol_daily = float(np.sqrt(w @ cov_daily @ w))
    return vol_daily * np.sqrt(252.0)


def save_allocation_chart(snapshot_df: pd.DataFrame, out_dir: Path = OUTPUT_DIR) -> Path:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.pie(
        snapshot_df["position_value_base"],
        labels=snapshot_df["ticker"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title(f"Portfolio Allocation ({BASE_CURRENCY})")
    filename = out_dir / f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return filename
