"""Feature engineering utilities for PortfolioDashboard."""
from __future__ import annotations

from typing import Iterable, Dict
import pandas as pd
import numpy as np


def calculate_momentum_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum metrics such as trailing returns and moving averages.

    Parameters
    ----------
    prices:
        DataFrame indexed by ``Date`` and ``Ticker`` with a ``Close`` column.

    Returns
    -------
    DataFrame
        DataFrame containing momentum features aligned with ``prices`` index.
    """
    if prices.empty:
        return pd.DataFrame(index=prices.index)

    df = prices.copy()
    df.sort_index(inplace=True)

    # 3-month and 12-month trailing returns
    df["ret_3m"] = df.groupby(level="Ticker")["Close"].pct_change(13)
    df["ret_12m"] = df.groupby(level="Ticker")["Close"].pct_change(52)

    # Simple moving average crossover indicator
    short_ma = df.groupby(level="Ticker")["Close"].transform(lambda x: x.rolling(4).mean())
    long_ma = df.groupby(level="Ticker")["Close"].transform(lambda x: x.rolling(26).mean())
    df["ma_cross"] = np.where(short_ma > long_ma, 1.0, 0.0)

    features = df[["ret_3m", "ret_12m", "ma_cross"]]
    return features


def lag_features(features: pd.DataFrame, n_lags: int = 1) -> pd.DataFrame:
    """Lag feature columns by ``n_lags`` periods to avoid lookahead bias."""
    lagged = features.groupby(level="Ticker").shift(n_lags)
    return lagged


def merge_features(prices: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Combine price and feature data into a single DataFrame."""
    df = pd.concat([prices, features], axis=1)
    df.dropna(inplace=True)
    return df

