"""Data loading utilities for PortfolioDashboard.

This module provides helper functions to fetch
historical weekly stock data and related metadata using
`yfinance`.
"""

from __future__ import annotations

from typing import Iterable, List, Dict
import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def load_weekly_prices(
    tickers: Iterable[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch weekly adjusted close prices for the given tickers.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols.
    start:
        Start date in ``YYYY-MM-DD`` format.
    end:
        End date in ``YYYY-MM-DD`` format.

    Returns
    -------
    DataFrame
        Multi-indexed by ``Date`` and ``Ticker`` with a single ``Close`` column.
    """
    tickers = list(tickers)
    logger.info("Downloading price data for %s tickers", len(tickers))
    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            interval="1wk",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:  # pragma: no cover - network errors
        logger.error("Failed to download prices: %s", exc)
        return pd.DataFrame(columns=["Close"], index=pd.MultiIndex.from_arrays([[], []], names=["Date", "Ticker"]))

    price_frames: List[pd.DataFrame] = []
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in data.columns.get_level_values(0):
                logger.warning("Ticker %s not found in downloaded data", ticker)
                continue
            df = data[ticker].copy()
            df["Ticker"] = ticker
            price_frames.append(df)
    else:
        # Single ticker case
        df = data.copy()
        df["Ticker"] = tickers[0]
        price_frames.append(df)

    prices = pd.concat(price_frames)
    prices.reset_index(inplace=True)
    prices.rename(columns={"index": "Date", "Adj Close": "Close"}, inplace=True)
    prices = prices[["Date", "Ticker", "Close"]]
    prices.sort_values(["Ticker", "Date"], inplace=True)
    prices.set_index(["Date", "Ticker"], inplace=True)
    return prices


def get_sector_info(tickers: Iterable[str]) -> Dict[str, str]:
    """Retrieve sector metadata for each ticker.

    This performs one request per ticker via ``yfinance.Ticker.info``.
    Missing sectors are filled with ``"Unknown"``.
    """
    sectors: Dict[str, str] = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            sector = info.get("sector") or "Unknown"
        except Exception as exc:  # pragma: no cover - network errors
            logger.error("Failed to fetch sector for %s: %s", t, exc)
            sector = "Unknown"
        sectors[t] = sector
    return sectors

