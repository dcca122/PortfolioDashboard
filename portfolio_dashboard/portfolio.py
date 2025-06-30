"""Position and weighting utilities for strategy backtesting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import pandas as pd


@dataclass
class PortfolioState:
    """Container for current portfolio positions."""

    capital: float
    positions: Dict[str, float] = field(default_factory=dict)  # weight by ticker


def equal_weight_signals(signals: pd.Series) -> Dict[str, float]:
    """Return equal-weight long/short weights from signal Series.

    Parameters
    ----------
    signals:
        Series of trading signals indexed by ticker. Positive for long,
        negative for short, zero for neutral.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping ticker to desired portfolio weight. Long and short
        books are each scaled to 50% gross exposure so that the net exposure is
        zero by construction.
    """
    longs = signals[signals > 0].index.tolist()
    shorts = signals[signals < 0].index.tolist()
    weights: Dict[str, float] = {}
    if longs:
        w = 0.5 / len(longs)
        for t in longs:
            weights[t] = w
    if shorts:
        w = -0.5 / len(shorts)
        for t in shorts:
            weights[t] = w
    return weights


def target_position_values(
    weights: Dict[str, float], capital: float, prices: pd.Series
) -> Dict[str, float]:
    """Convert weight targets to notional dollar positions."""
    return {t: capital * w for t, w in weights.items() if t in prices.index}
