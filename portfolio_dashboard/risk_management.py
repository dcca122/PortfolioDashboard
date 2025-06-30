"""Risk management helpers for the backtesting engine."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def volatility_target(
    returns: pd.Series, target_vol: float, lookback: int = 52
) -> float:
    """Return scaling factor so that annualized vol matches ``target_vol``."""
    if returns.empty:
        return 1.0
    recent = returns.dropna().iloc[-lookback:]
    if recent.empty:
        return 1.0
    vol = recent.std() * np.sqrt(52)
    if vol == 0:
        return 1.0
    return float(target_vol / vol)


def sector_neutral_weights(
    weights: Dict[str, float], sectors: Dict[str, str]
) -> Dict[str, float]:
    """Adjust weights to have zero net sector exposure."""
    if not weights:
        return weights
    df = pd.DataFrame({'weight': weights})
    df['sector'] = df.index.map(sectors)
    sector_totals = df.groupby('sector')['weight'].sum()
    for sector, total in sector_totals.items():
        tickers = df[df['sector'] == sector].index
        if len(tickers) == 0:
            continue
        adj = total / len(tickers)
        for t in tickers:
            weights[t] -= adj
    return weights


def stop_loss_signals(
    current_prices: pd.Series,
    entry_prices: Dict[str, float],
    weights: Dict[str, float],
    stop_pct: float,
) -> Dict[str, float]:
    """Close positions breaching the stop loss threshold."""
    for t, entry in list(entry_prices.items()):
        if t not in current_prices.index or t not in weights:
            continue
        price = current_prices[t]
        w = weights[t]
        if w == 0:
            continue
        ret = (price - entry) / entry
        if w < 0:
            ret = -ret
        if ret <= -stop_pct:
            weights[t] = 0.0
            entry_prices.pop(t, None)
    return weights
