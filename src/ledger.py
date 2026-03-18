from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from src.config import BASE_CURRENCY


TRANSACTION_COLUMNS = [
    "date",
    "ticker",
    "exchange",
    "side",
    "quantity",
    "price",
    "currency",
    "fee",
    "notes",
]


@dataclass
class LedgerReport:
    positions: pd.DataFrame
    totals: Dict[str, float]


def normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TRANSACTION_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[TRANSACTION_COLUMNS]

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out["ticker"].fillna("").astype(str).str.strip().str.upper()
    out["exchange"] = out["exchange"].fillna("").astype(str).str.strip().str.upper()
    out["side"] = out["side"].fillna("").astype(str).str.strip().str.lower()
    out["currency"] = out["currency"].fillna("").astype(str).str.strip().str.upper()
    out["notes"] = out["notes"].fillna("").astype(str).str.strip()

    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["fee"] = pd.to_numeric(out["fee"], errors="coerce").fillna(0.0)

    out = out.dropna(subset=["date", "quantity", "price"])
    out = out[(out["ticker"] != "") & (out["currency"] != "")]
    out = out[out["side"].isin(["buy", "sell"])]
    out = out[(out["quantity"] > 0) & (out["price"] >= 0) & (out["fee"] >= 0)]
    out = out.sort_values("date").reset_index(drop=True)
    return out


def build_ledger_report(
    transactions_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    fx_rates: Dict[tuple[str, str], float],
) -> LedgerReport:
    tx = normalize_transactions(transactions_df)
    if tx.empty:
        empty = pd.DataFrame(
            columns=[
                "ticker",
                "exchange",
                "currency",
                "open_quantity",
                "avg_cost_local",
                "market_price",
                "realized_pnl_base",
                "unrealized_pnl_base",
                "total_pnl_base",
                "market_value_base",
            ]
        )
        return LedgerReport(
            positions=empty,
            totals={
                "realized_pnl_base": 0.0,
                "unrealized_pnl_base": 0.0,
                "total_pnl_base": 0.0,
                "market_value_base": 0.0,
            },
        )

    state: dict[str, dict[str, float | str]] = {}
    for _, row in tx.iterrows():
        ticker = str(row["ticker"])
        side = str(row["side"])
        qty = float(row["quantity"])
        price = float(row["price"])
        fee = float(row["fee"])
        currency = str(row["currency"])
        exchange = str(row["exchange"])

        if ticker not in state:
            state[ticker] = {
                "exchange": exchange,
                "currency": currency,
                "open_quantity": 0.0,
                "avg_cost_local": 0.0,
                "realized_pnl_local": 0.0,
            }

        s = state[ticker]
        open_qty = float(s["open_quantity"])
        avg_cost = float(s["avg_cost_local"])
        realized = float(s["realized_pnl_local"])

        if side == "buy":
            new_qty = open_qty + qty
            total_cost = open_qty * avg_cost + qty * price + fee
            new_avg = total_cost / new_qty if new_qty > 0 else 0.0
            s["open_quantity"] = new_qty
            s["avg_cost_local"] = new_avg
            s["realized_pnl_local"] = realized
        elif side == "sell":
            if qty > open_qty + 1e-12:
                raise ValueError(
                    f"Invalid sell for {ticker}: sell quantity {qty} exceeds open quantity {open_qty}."
                )
            proceeds = qty * price - fee
            cogs = qty * avg_cost
            realized += proceeds - cogs
            s["open_quantity"] = max(0.0, open_qty - qty)
            s["avg_cost_local"] = avg_cost if s["open_quantity"] > 0 else 0.0
            s["realized_pnl_local"] = realized

    out = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "exchange": str(v["exchange"]),
                "currency": str(v["currency"]),
                "open_quantity": float(v["open_quantity"]),
                "avg_cost_local": float(v["avg_cost_local"]),
                "realized_pnl_local": float(v["realized_pnl_local"]),
            }
            for ticker, v in state.items()
        ]
    )

    snapshot_view = snapshot_df[
        ["ticker", "market_price", "fx_to_base", "position_value_base", "position_value_local"]
    ].copy()
    snapshot_view = snapshot_view.drop_duplicates(subset=["ticker"])
    out = out.merge(snapshot_view, on="ticker", how="left")

    def fx_to_base(row: pd.Series) -> float:
        if pd.notna(row.get("fx_to_base")):
            return float(row["fx_to_base"])
        ccy = str(row["currency"]).upper().strip()
        if ccy == BASE_CURRENCY:
            return 1.0
        return float(fx_rates.get((ccy, BASE_CURRENCY), np.nan))

    out["fx_to_base"] = out.apply(fx_to_base, axis=1)
    out["market_price"] = pd.to_numeric(out["market_price"], errors="coerce").fillna(0.0)
    out["realized_pnl_base"] = out["realized_pnl_local"] * out["fx_to_base"]
    out["market_value_base"] = out["open_quantity"] * out["market_price"] * out["fx_to_base"]
    out["cost_basis_base"] = out["open_quantity"] * out["avg_cost_local"] * out["fx_to_base"]
    out["unrealized_pnl_base"] = out["market_value_base"] - out["cost_basis_base"]
    out["total_pnl_base"] = out["realized_pnl_base"] + out["unrealized_pnl_base"]
    out = out.sort_values("total_pnl_base", ascending=False).reset_index(drop=True)

    totals = {
        "realized_pnl_base": float(out["realized_pnl_base"].sum()),
        "unrealized_pnl_base": float(out["unrealized_pnl_base"].sum()),
        "total_pnl_base": float(out["total_pnl_base"].sum()),
        "market_value_base": float(out["market_value_base"].sum()),
    }
    return LedgerReport(positions=out, totals=totals)
