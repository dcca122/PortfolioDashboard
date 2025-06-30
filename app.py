"""Unified Streamlit dashboard for strategy results and options analytics."""
from __future__ import annotations

import datetime as dt
from typing import List

import pandas as pd
import streamlit as st

from portfolio_dashboard.backtest import run_backtest
from portfolio_dashboard.data_loader import load_weekly_prices, get_sector_info
from portfolio_dashboard.feature_engineering import calculate_momentum_features, lag_features
from portfolio_dashboard.visualization import (
    plot_equity_and_drawdown,
    plot_rolling_metrics,
    show_summary_stats,
)
from portfolio_dashboard.options_dashboard import main as options_ui


@st.cache_data(show_spinner=False)
def fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    prices = load_weekly_prices(tickers, start, end)
    return prices['Close'].unstack('Ticker')


@st.cache_data(show_spinner=False)
def fetch_sectors(tickers: List[str]):
    return get_sector_info(tickers)


def generate_signals(features: pd.DataFrame, upper_q: float, lower_q: float) -> pd.DataFrame:
    momentum = features['ret_12m']
    ranks = momentum.groupby('Date').rank(pct=True)
    sig = pd.Series(0, index=ranks.index, dtype=float)
    sig[ranks >= upper_q] = 1
    sig[ranks <= lower_q] = -1
    return sig.unstack('Ticker').fillna(0)


def run_momentum_backtest(
    tickers: List[str],
    start: str,
    end: str,
    upper_q: float,
    lower_q: float,
    target_vol: float,
    stop_loss: float,
):
    prices = fetch_prices(tickers, start, end)
    sectors = fetch_sectors(tickers)

    feats = calculate_momentum_features(
        prices.stack().to_frame(name="Close")
    )
    feats = lag_features(feats)
    signals = generate_signals(feats, upper_q, lower_q)

    result = run_backtest(
        prices,
        signals,
        sectors,
        target_vol=target_vol,
        stop_loss_pct=stop_loss,
    )
    return result


def strategy_section():
    st.header("Momentum Strategy Backtest")
    tickers = st.text_input("Tickers (comma separated)", "AAPL,MSFT,GOOGL")
    col1, col2 = st.columns(2)
    start = col1.date_input("Start", dt.date(2020, 1, 1))
    end = col2.date_input("End", dt.date.today())
    upper_q = st.slider("Long quantile", 0.5, 1.0, 0.8, step=0.05)
    lower_q = st.slider("Short quantile", 0.0, 0.5, 0.2, step=0.05)
    target_vol = st.slider("Target volatility", 0.05, 0.3, 0.1, step=0.01)
    stop_loss = st.slider("Stop loss %", 0.05, 0.3, 0.1, step=0.01)

    if st.button("Run Backtest"):
        tlist = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        result = run_momentum_backtest(
            tlist,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            upper_q,
            lower_q,
            target_vol,
            stop_loss,
        )
        show_summary_stats(result)
        plot_equity_and_drawdown(result)
        plot_rolling_metrics(result)


def main():
    st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
    page = st.sidebar.selectbox("Section", ["Backtest", "Options"])

    if page == "Backtest":
        strategy_section()
    else:
        options_ui()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
