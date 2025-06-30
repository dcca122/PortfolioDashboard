"""Vollib-powered option pricing and Greek calculations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from vollib.black_scholes import black_scholes
from vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega, rho
from vollib.black_scholes.implied_volatility import implied_volatility

import pandas as pd
import yfinance as yf


@dataclass
class OptionGreeks:
    """Container for standard option Greeks."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


# ---------------------------------------------------------------------------
# Core option analytics wrappers
# ---------------------------------------------------------------------------

def black_scholes_price(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> float:
    """Return Black-Scholes price via ``vollib``."""
    flag = "c" if option_type.lower().startswith("c") else "p"
    return float(black_scholes(flag, spot, strike, t, r, sigma, q))


def option_greeks(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> OptionGreeks:
    """Return Delta, Gamma, Theta, Vega and Rho via ``vollib``."""
    flag = "c" if option_type.lower().startswith("c") else "p"
    return OptionGreeks(
        float(delta(flag, spot, strike, t, r, sigma, q)),
        float(gamma(flag, spot, strike, t, r, sigma, q)),
        float(theta(flag, spot, strike, t, r, sigma, q)),
        float(vega(flag, spot, strike, t, r, sigma, q)),
        float(rho(flag, spot, strike, t, r, sigma, q)),
    )


def implied_vol(
    market_price: float,
    spot: float,
    strike: float,
    t: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
) -> float:
    """Solve for the implied volatility from a market price."""
    flag = "c" if option_type.lower().startswith("c") else "p"
    try:
        vol = implied_volatility(market_price, flag, spot, strike, t, r, q)
    except Exception:
        vol = float("nan")
    return float(vol)


def compute_all_option_metrics(
    flag: str,
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    market_price: float | None = None,
    q: float = 0.0,
) -> Dict[str, Any]:
    """Return fair value, Greeks and optionally implied volatility."""
    price = black_scholes(flag, s, k, t, r, sigma, q)
    greeks = {
        "delta": delta(flag, s, k, t, r, sigma, q),
        "gamma": gamma(flag, s, k, t, r, sigma, q),
        "theta": theta(flag, s, k, t, r, sigma, q),
        "vega": vega(flag, s, k, t, r, sigma, q),
        "rho": rho(flag, s, k, t, r, sigma, q),
    }
    out: Dict[str, Any] = {"fair_value": float(price), "greeks": {k: float(v) for k, v in greeks.items()}}
    if market_price is not None:
        try:
            out["implied_vol"] = float(
                implied_volatility(market_price, flag, s, k, t, r, q)
            )
        except Exception:
            out["implied_vol"] = None
    return out


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_option_chain(ticker: str, expiry: str) -> pd.DataFrame:
    """Fetch option chain for ``ticker`` and ``expiry`` from Yahoo! Finance."""
    data = yf.Ticker(ticker)
    chain = data.option_chain(expiry)
    calls = chain.calls.assign(option_type="call")
    puts = chain.puts.assign(option_type="put")
    return pd.concat([calls, puts])


__all__ = [
    "OptionGreeks",
    "black_scholes_price",
    "option_greeks",
    "implied_vol",
    "compute_all_option_metrics",
    "load_option_chain",
]
