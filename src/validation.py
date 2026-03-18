from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationReport:
    errors: list[str]
    warnings: list[str]

    @property
    def is_clean(self) -> bool:
        return not self.errors and not self.warnings


def validate_portfolio_inputs(portfolio: pd.DataFrame) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    p = portfolio.copy()
    p["ticker"] = p["ticker"].astype(str).str.strip().str.upper()
    p["price_source"] = p["price_source"].astype(str).str.strip().str.lower()
    p["manual_price"] = pd.to_numeric(p["manual_price"], errors="coerce")
    p["avg_cost"] = pd.to_numeric(p["avg_cost"], errors="coerce")
    p["shares"] = pd.to_numeric(p["shares"], errors="coerce")

    if p["ticker"].duplicated().any():
        dups = sorted(p.loc[p["ticker"].duplicated(), "ticker"].unique().tolist())
        warnings.append(f"Duplicate tickers found in portfolio: {dups}")

    missing_avg = p[p["avg_cost"].isna() | (p["avg_cost"] <= 0)]
    if not missing_avg.empty:
        warnings.append(
            f"{len(missing_avg)} position(s) have missing/zero avg_cost (P&L/return quality will be weak)."
        )

    invalid_shares = p[p["shares"].isna() | (p["shares"] <= 0)]
    if not invalid_shares.empty:
        errors.append(f"{len(invalid_shares)} position(s) have invalid shares (must be > 0).")

    missing_yf_symbol = p[(p["price_source"] == "yfinance") & (p["symbol_yf"].fillna("").astype(str).str.strip() == "")]
    if not missing_yf_symbol.empty:
        errors.append(
            f"{len(missing_yf_symbol)} position(s) use yfinance but have empty symbol_yf."
        )

    manual_no_price = p[(p["price_source"] == "manual") & (p["manual_price"].isna() | (p["manual_price"] <= 0))]
    if not manual_no_price.empty:
        errors.append(
            f"{len(manual_no_price)} manual-priced position(s) missing manual_price."
        )

    return ValidationReport(errors=errors, warnings=warnings)


def validate_transactions_inputs(transactions: pd.DataFrame) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    tx = transactions.copy()
    tx["ticker"] = tx["ticker"].fillna("").astype(str).str.strip().str.upper()
    tx["side"] = tx["side"].fillna("").astype(str).str.strip().str.lower()
    tx["quantity"] = pd.to_numeric(tx["quantity"], errors="coerce")
    tx["price"] = pd.to_numeric(tx["price"], errors="coerce")
    tx["fee"] = pd.to_numeric(tx["fee"], errors="coerce").fillna(0.0)
    tx["date"] = pd.to_datetime(tx["date"], errors="coerce")

    if tx.empty:
        warnings.append("No transactions found yet.")
        return ValidationReport(errors=errors, warnings=warnings)

    bad_rows = tx[
        tx["date"].isna()
        | (tx["ticker"] == "")
        | (~tx["side"].isin(["buy", "sell"]))
        | (tx["quantity"].isna())
        | (tx["quantity"] <= 0)
        | (tx["price"].isna())
        | (tx["price"] < 0)
        | (tx["fee"] < 0)
    ]
    if not bad_rows.empty:
        errors.append(f"{len(bad_rows)} transaction row(s) have invalid fields.")

    # Quick integrity check: cannot net-sell more than net-buys per ticker.
    signed = tx.copy()
    signed["signed_qty"] = signed["quantity"].where(signed["side"] == "buy", -signed["quantity"])
    net = signed.groupby("ticker", dropna=True)["signed_qty"].sum()
    over_sold = net[net < -1e-12]
    if not over_sold.empty:
        errors.append(
            f"Net negative quantity found for ticker(s): {sorted(over_sold.index.tolist())}."
        )

    return ValidationReport(errors=errors, warnings=warnings)
