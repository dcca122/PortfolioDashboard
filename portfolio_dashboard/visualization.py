"""Plotly-based visualization helpers for strategy results."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .backtest import BacktestResult


def plot_equity_and_drawdown(result: BacktestResult) -> None:
    """Display equity curve and drawdown charts."""
    equity = result.equity_curve
    drawdown = equity / equity.cummax() - 1

    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Equity"))
    eq_fig.update_layout(title="Equity Curve", yaxis_title="Equity")

    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", name="Drawdown"))
    dd_fig.update_layout(title="Drawdown", yaxis_title="Drawdown", yaxis_tickformat=".0%")

    st.plotly_chart(eq_fig, use_container_width=True)
    st.plotly_chart(dd_fig, use_container_width=True)


def plot_rolling_metrics(result: BacktestResult, window: int = 26) -> None:
    """Display rolling Sharpe ratio and volatility."""
    ret = result.equity_curve.pct_change()
    roll_vol = ret.rolling(window).std() * np.sqrt(52)
    roll_sharpe = ret.rolling(window).mean() / ret.rolling(window).std() * np.sqrt(52)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe, name="Rolling Sharpe"))
    fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol, name="Rolling Volatility"))
    fig.update_layout(title="Rolling Metrics")
    st.plotly_chart(fig, use_container_width=True)


def show_summary_stats(result: BacktestResult) -> None:
    """Display key statistics from a ``BacktestResult``."""
    stats = result.stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe", f"{stats.get('sharpe', float('nan')):.2f}")
    col2.metric("Max Drawdown", f"{stats.get('max_drawdown', float('nan')):.2%}")
    col3.metric("Turnover", f"{stats.get('turnover', float('nan')):.2f}")
