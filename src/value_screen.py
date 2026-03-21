"""
Simple value-style ranking from Yahoo Finance fundamentals.

Not financial advice. Ratios are noisy; use as a screen + your policy, not blind picks.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Optional

import pandas as pd
import yfinance as yf


def fetch_value_metrics(symbol: str) -> dict:
    symbol = str(symbol).strip()
    if not symbol:
        return {}

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        info = yf.Ticker(symbol).info or {}

    return {
        "symbol_yf": symbol,
        "name": info.get("shortName") or info.get("longName") or "",
        "sector": info.get("sector") or "",
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "price_to_book": info.get("priceToBook"),
        "market_cap": info.get("marketCap"),
    }


def _finite_positive(x: Optional[float]) -> bool:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    try:
        return float(x) > 0
    except (TypeError, ValueError):
        return False


def build_value_table(entries: list[tuple[str, str]]) -> pd.DataFrame:
    """
    entries: list of (your_ticker_label, yahoo_symbol), e.g. [("GOOGL", "GOOGL")].
    """
    rows = []
    for ticker, sym in entries:
        sym = str(sym).strip()
        ticker = str(ticker).strip()
        if not sym or sym.lower() == "nan":
            continue
        m = fetch_value_metrics(sym)
        if not m:
            rows.append({"ticker": ticker, "symbol_yf": sym, "error": "no_data"})
            continue
        rows.append({"ticker": ticker, **m, "error": ""})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Simple combined rank: lower P/E and lower P/B = more "value" among this batch.
    # Only rank rows with positive trailing P/E and P/B (classic cheapness screen).
    work = df.copy()
    work["trailing_pe"] = pd.to_numeric(work["trailing_pe"], errors="coerce")
    work["price_to_book"] = pd.to_numeric(work["price_to_book"], errors="coerce")
    work["value_score"] = pd.NA

    mask = work["trailing_pe"].apply(_finite_positive) & work["price_to_book"].apply(_finite_positive)
    if mask.any():
        work.loc[mask, "pe_pct"] = work.loc[mask, "trailing_pe"].rank(pct=True, ascending=True)
        work.loc[mask, "pb_pct"] = work.loc[mask, "price_to_book"].rank(pct=True, ascending=True)
        work.loc[mask, "value_score"] = (
            (1.0 - work.loc[mask, "pe_pct"]) * 50.0 + (1.0 - work.loc[mask, "pb_pct"]) * 50.0
        )

    return work.sort_values(
        ["value_score", "trailing_pe"],
        ascending=[False, True],
        na_position="last",
    )
