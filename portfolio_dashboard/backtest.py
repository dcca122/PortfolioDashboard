"""Simple backtesting engine for weekly long/short strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .portfolio import equal_weight_signals
from .risk_management import (
    sector_neutral_weights,
    volatility_target,
    stop_loss_signals,
)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    stats: Dict[str, float]


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    sectors: Dict[str, str],
    target_vol: float = 0.1,
    transaction_cost: float = 0.001,
    stop_loss_pct: float = 0.1,
) -> BacktestResult:
    """Run a vectorized backtest with basic risk controls."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)
    signals = signals.reindex(prices.index).fillna(0)

    returns = prices.pct_change().dropna()
    tickers = returns.columns

    weights = pd.DataFrame(index=signals.index, columns=tickers, data=0.0)
    entry_prices: Dict[str, float] = {}

    equity = [1.0]
    trade_records = []
    prev_weights = pd.Series(0.0, index=tickers)

    for i in range(len(returns.index) - 1):
        date = returns.index[i]
        next_date = returns.index[i + 1]
        signal = signals.loc[date]
        w = equal_weight_signals(signal)
        w = sector_neutral_weights(w, sectors)
        w = pd.Series(w, index=tickers).fillna(0.0)

        # stop loss based on previous week's close
        w = stop_loss_signals(prices.loc[date], entry_prices, w.to_dict(), stop_loss_pct)
        w = pd.Series(w, index=tickers).fillna(0.0)

        hist_rets = pd.Series(equity).pct_change().dropna()
        scale = volatility_target(hist_rets, target_vol) if len(hist_rets) > 5 else 1.0
        w = w * scale

        weights.loc[date] = w
        turnover = (w - prev_weights).abs().sum()
        trade_records.append({'Date': date, 'turnover': turnover})

        port_ret = (prev_weights * returns.loc[date]).sum()
        new_equity = equity[-1] * (1 + port_ret) - turnover * transaction_cost
        equity.append(new_equity)

        # update entry prices for new positions
        for t in tickers:
            if prev_weights[t] == 0 and w[t] != 0:
                entry_prices[t] = prices.loc[date, t]
            if w[t] == 0:
                entry_prices.pop(t, None)

        prev_weights = w

    equity_curve = pd.Series(equity[1:], index=returns.index[:-1])
    stats = {
        'sharpe': float(np.sqrt(52) * equity_curve.pct_change().mean() / equity_curve.pct_change().std()),
        'max_drawdown': float((equity_curve.cummax() - equity_curve).max() / equity_curve.cummax().max()),
        'turnover': float(np.mean([r['turnover'] for r in trade_records])),
    }
    trades = pd.DataFrame(trade_records)
    return BacktestResult(equity_curve, trades, stats)
