"""End-to-end data pipeline for PortfolioDashboard.

This script demonstrates how to download weekly stock prices,
compute momentum features, build a sector graph, and export
a merged feature set for modeling.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd

from .data_loader import load_weekly_prices, get_sector_info
from .feature_engineering import calculate_momentum_features, lag_features, merge_features
from .graph_builder import build_sector_graph, compute_graph_features


def build_feature_dataset(
    tickers: Iterable[str],
    start: str,
    end: str,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Run the complete pipeline and return the feature DataFrame."""
    prices = load_weekly_prices(tickers, start, end)
    sectors = get_sector_info(tickers)
    momentum = calculate_momentum_features(prices)
    momentum = lag_features(momentum, n_lags=1)

    sector_graph = build_sector_graph(sectors)
    g_features = compute_graph_features(sector_graph)

    # merge graph features with momentum features
    full = merge_features(prices, momentum)
    full = full.reset_index().merge(g_features, left_on="Ticker", right_index=True, how="left")
    full.set_index(["Date", "Ticker"], inplace=True)

    if output_path:
        if output_path.endswith(".csv"):
            full.to_csv(output_path)
        else:
            full.to_parquet(output_path)
    return full


if __name__ == "__main__":  # pragma: no cover - manual execution
    SAMPLE_TICKERS = ["AAPL", "MSFT", "GOOGL"]
    START_DATE = "2020-01-01"
    END_DATE = datetime.today().strftime("%Y-%m-%d")
    df = build_feature_dataset(SAMPLE_TICKERS, START_DATE, END_DATE)
    print(df.head())

