"""Streamlit app for single-contract options analysis."""
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

from .options_utils import black_scholes_price, option_greeks, implied_volatility


def load_option_chain(ticker: str, expiry: str) -> pd.DataFrame:
    """Fetch option chain for ticker and expiry."""
    data = yf.Ticker(ticker)
    chain = data.option_chain(expiry)
    calls = chain.calls.assign(option_type='call')
    puts = chain.puts.assign(option_type='put')
    return pd.concat([calls, puts])


def main() -> None:
    st.title("Options Valuation")

    ticker = st.text_input("Ticker", "AAPL")
    if not ticker:
        st.stop()

    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        st.error("No option expiries found")
        st.stop()

    expiry = st.selectbox("Expiry", expiries)
    chain = load_option_chain(ticker, expiry)

    option_type = st.selectbox("Call/Put", ["call", "put"])
    strikes = chain.loc[chain["option_type"] == option_type, "strike"].unique()
    strike = st.selectbox("Strike", sorted(strikes))

    row = chain[(chain["option_type"] == option_type) & (chain["strike"] == strike)].iloc[0]
    market_price = row["lastPrice"]
    iv_chain = row["impliedVolatility"]

    spot = tk.history(period="1d")["Close"][0]
    T = (pd.to_datetime(expiry) - pd.Timestamp.today()).days / 365

    r = st.number_input("Risk-free rate", value=0.03)
    q = st.number_input("Dividend yield", value=0.0)
    iv = st.number_input("Implied volatility", value=float(iv_chain))

    bs_price = black_scholes_price(spot, strike, T, r, iv, q, option_type)
    greeks = option_greeks(spot, strike, T, r, iv, q, option_type)

    st.subheader("Results")
    st.write("Market price:", market_price)
    st.write("Black-Scholes price:", round(bs_price, 4))
    st.write(greeks)

    user_price = st.number_input("Input market price to solve IV", value=float(market_price))
    if user_price:
        iv_est = implied_volatility(user_price, spot, strike, T, r, q, option_type)
        if not np.isnan(iv_est):
            st.write("Implied volatility from price:", round(iv_est, 4))
        else:
            st.write("Could not solve for implied volatility")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
