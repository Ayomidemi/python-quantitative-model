from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd
import yfinance as yf

from src.config import DATA_DIR, FX_PAIRS


@dataclass
class MarketSnapshot:
    prices: Dict[str, float]
    fx_rates: Dict[tuple[str, str], float]


def load_portfolio(path: Optional[str] = None) -> pd.DataFrame:
    csv_path = DATA_DIR / "portfolio.csv" if path is None else path
    df = pd.read_csv(csv_path)
    required_cols = {
        "ticker",
        "exchange",
        "shares",
        "avg_cost",
        "currency",
        "price_source",
        "symbol_yf",
        "manual_price",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"portfolio.csv missing columns: {sorted(missing)}")
    return df


def load_watchlist(path: Optional[str] = None) -> pd.DataFrame:
    csv_path = DATA_DIR / "watchlist.csv" if path is None else path
    return pd.read_csv(csv_path)


def load_transactions(path: Optional[str] = None) -> pd.DataFrame:
    csv_path = DATA_DIR / "transactions.csv" if path is None else path
    df = pd.read_csv(csv_path)
    required_cols = {
        "date",
        "ticker",
        "exchange",
        "side",
        "quantity",
        "price",
        "currency",
        "fee",
        "notes",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"transactions.csv missing columns: {sorted(missing)}")
    return df


def _last_close(symbol: str, period: str = "7d") -> Optional[float]:
    if pd.isna(symbol):
        return None
    symbol = str(symbol).strip()
    if not symbol or symbol.lower() == "nan":
        return None
    # Silence noisy provider logs for unsupported symbols.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        history = yf.Ticker(symbol).history(period=period)
    if history.empty:
        return None
    close = history["Close"].dropna()
    if close.empty:
        return None
    return float(close.iloc[-1])


def fetch_prices(portfolio_df: pd.DataFrame) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for _, row in portfolio_df.iterrows():
        ticker = str(row["ticker"]).strip()
        source = str(row["price_source"]).lower().strip()
        manual_price = row.get("manual_price")
        symbol = row.get("symbol_yf")

        fetched_price: Optional[float] = None
        if source == "yfinance":
            fetched_price = _last_close(str(symbol))
        elif source == "manual":
            fetched_price = float(manual_price) if pd.notna(manual_price) else None
        else:
            raise ValueError(f"Unknown price_source '{source}' for ticker '{ticker}'.")

        if fetched_price is None:
            # Fallback to manual price if online fetch is unavailable.
            if pd.notna(manual_price):
                fetched_price = float(manual_price)
            else:
                raise ValueError(
                    f"Could not resolve price for {ticker}. "
                    "Set a manual_price or valid symbol_yf."
                )
        prices[ticker] = fetched_price
    return prices


def fetch_fx_rates(pairs: Iterable[tuple[str, str]]) -> Dict[tuple[str, str], float]:
    rates: Dict[tuple[str, str], float] = {}
    for base, quote in pairs:
        if base == quote:
            rates[(base, quote)] = 1.0
            continue
        symbol = FX_PAIRS.get((base, quote))
        if symbol is None:
            raise ValueError(f"No FX symbol configured for {base}/{quote}.")
        fx = _last_close(symbol)
        if fx is None:
            raise ValueError(f"Unable to fetch FX rate for {base}/{quote} ({symbol}).")
        rates[(base, quote)] = fx
    return rates
