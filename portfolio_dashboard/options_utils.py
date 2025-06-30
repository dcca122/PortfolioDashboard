"""Black-Scholes pricing utilities and Greek calculations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


@dataclass
class OptionGreeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def _d1(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * np.sqrt(T)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> float:
    """Return the Black-Scholes price for a call or put option."""
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(d1, sigma, T)
    if option_type == "call":
        price = np.exp(-q * T) * S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - np.exp(-q * T) * S * norm.cdf(-d1)
    return float(price)


def option_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> OptionGreeks:
    """Return Delta, Gamma, Theta, Vega, Rho as an ``OptionGreeks`` object."""
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(d1, sigma, T)
    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        )
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
            -S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) * np.exp(-q * T) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    return OptionGreeks(float(delta), float(gamma), float(theta), float(vega), float(rho))


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    max_iter: int = 100,
) -> float:
    """Solve for the implied volatility from a market price."""

    def objective(sigma: float) -> float:
        return black_scholes_price(S, K, T, r, sigma, q, option_type) - market_price

    try:
        vol = brentq(objective, 1e-6, 5.0, maxiter=max_iter)
    except Exception:
        vol = np.nan
    return float(vol)
