"""Option pricing and Greek calculations using pure Python formulas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

import pandas as pd
import yfinance as yf


def black_scholes_price(S, K, T, r, sigma, option_type='c'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'c':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'p':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'c' (call) or 'p' (put)")
    return price

# Refactored June 2025: Removed vollib/lets_be_rational, now uses pure Python Black-Scholes


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

def _d1_d2(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> tuple[float, float]:
    """Return ``d1`` and ``d2`` for the Black-Scholes model."""
    d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * t) / (
        sigma * np.sqrt(t)
    )
    d2 = d1 - sigma * np.sqrt(t)
    return d1, d2


def option_greeks(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> OptionGreeks:
    """Return Delta, Gamma, Theta, Vega and Rho via analytic formulas."""
    flag = "c" if option_type.lower().startswith("c") else "p"
    d1, d2 = _d1_d2(spot, strike, t, r, sigma, q)
    pdf_d1 = norm.pdf(d1)
    if flag == "c":
        delta_v = np.exp(-q * t) * norm.cdf(d1)
        theta_v = (
            -spot * sigma * np.exp(-q * t) * pdf_d1 / (2 * np.sqrt(t))
            - r * strike * np.exp(-r * t) * norm.cdf(d2)
            + q * spot * np.exp(-q * t) * norm.cdf(d1)
        )
        rho_v = strike * t * np.exp(-r * t) * norm.cdf(d2)
    else:
        delta_v = np.exp(-q * t) * (norm.cdf(d1) - 1)
        theta_v = (
            -spot * sigma * np.exp(-q * t) * pdf_d1 / (2 * np.sqrt(t))
            + r * strike * np.exp(-r * t) * norm.cdf(-d2)
            - q * spot * np.exp(-q * t) * norm.cdf(-d1)
        )
        rho_v = -strike * t * np.exp(-r * t) * norm.cdf(-d2)

    gamma_v = np.exp(-q * t) * pdf_d1 / (spot * sigma * np.sqrt(t))
    vega_v = spot * np.exp(-q * t) * np.sqrt(t) * pdf_d1

    return OptionGreeks(
        float(delta_v),
        float(gamma_v),
        float(theta_v),
        float(vega_v),
        float(rho_v),
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

    def objective(sigma: float) -> float:
        price = np.exp(-q * t) * black_scholes_price(spot, strike, t, r - q, sigma, flag)
        return price - market_price

    try:
        vol = brentq(objective, 1e-6, 5.0)
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
    price = np.exp(-q * t) * black_scholes_price(s, k, t, r - q, sigma, flag)

    greeks_obj = option_greeks(s, k, t, r, sigma, q, option_type="call" if flag == "c" else "put")
    greeks = {
        "delta": greeks_obj.delta,
        "gamma": greeks_obj.gamma,
        "theta": greeks_obj.theta,
        "vega": greeks_obj.vega,
        "rho": greeks_obj.rho,
    }

    out: Dict[str, Any] = {
        "fair_value": float(price),
        "greeks": {k: float(v) for k, v in greeks.items()},
    }

    if market_price is not None:
        try:
            out["implied_vol"] = float(
                implied_vol(market_price, s, k, t, r, q, "call" if flag == "c" else "put")
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
