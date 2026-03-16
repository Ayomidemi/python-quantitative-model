from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MonteCarloResult:
    expected_final_value: float
    p5_final_value: float
    p50_final_value: float
    p95_final_value: float
    prob_loss: float


def simulate_portfolio_paths(
    initial_value: float,
    annual_return: float,
    annual_volatility: float,
    horizon_days: int = 63,
    num_sims: int = 5000,
    random_seed: int = 42,
) -> MonteCarloResult:
    if initial_value <= 0:
        raise ValueError("initial_value must be positive.")

    rng = np.random.default_rng(random_seed)
    dt = 1.0 / 252.0
    mu = annual_return
    sigma = max(annual_volatility, 1e-8)

    shocks = rng.normal(
        loc=(mu - 0.5 * sigma * sigma) * dt,
        scale=sigma * np.sqrt(dt),
        size=(num_sims, horizon_days),
    )
    growth = np.exp(shocks).prod(axis=1)
    finals = initial_value * growth

    return MonteCarloResult(
        expected_final_value=float(finals.mean()),
        p5_final_value=float(np.percentile(finals, 5)),
        p50_final_value=float(np.percentile(finals, 50)),
        p95_final_value=float(np.percentile(finals, 95)),
        prob_loss=float((finals < initial_value).mean()),
    )
