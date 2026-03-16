from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_returns(symbol: str, period: str = "2y") -> pd.Series:
    history = yf.Ticker(symbol).history(period=period)
    if history.empty:
        return pd.Series(dtype=float)
    return history["Close"].pct_change().dropna()


def momentum_signal(returns: pd.Series, lookback_days: int = 63) -> Optional[float]:
    if len(returns) < lookback_days:
        return None
    window = returns.iloc[-lookback_days:]
    return float((1.0 + window).prod() - 1.0)


def rolling_mean_forecast(returns: pd.Series, lookback_days: int = 30) -> Optional[float]:
    if len(returns) < lookback_days:
        return None
    return float(returns.iloc[-lookback_days:].mean())


def garch_vol_forecast(returns: pd.Series, horizon_days: int = 21) -> Optional[float]:
    if len(returns) < 120:
        return None

    try:
        from arch import arch_model
    except Exception:
        return None

    # Convert to percentage returns for more stable GARCH fitting.
    model = arch_model(returns * 100.0, p=1, q=1, vol="GARCH", dist="normal")
    fitted = model.fit(disp="off")
    forecast = fitted.forecast(horizon=horizon_days, reindex=False)
    daily_var_pct2 = float(forecast.variance.values[-1].mean())
    daily_vol = np.sqrt(daily_var_pct2) / 100.0
    return daily_vol * np.sqrt(252.0)
