"""
ffcapm.data
--------------
Fetch stock prices and Fama-French factor data.
"""

from __future__ import annotations

import warnings
from typing import Literal

import pandas as pd
import yfinance as yf
import pandas_datareader.data as web


_FF_DATASETS = {
    "daily":   "F-F_Research_Data_5_Factors_2x3_daily",
    "monthly": "F-F_Research_Data_5_Factors_2x3",
}

_FF3_COLS = ["Mkt-RF", "SMB", "HML", "RF"]
_FF5_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]


def fetch_prices(
    ticker: str,
    start: str,
    end: str,
    frequency: Literal["daily", "monthly"] = "daily",
) -> pd.DataFrame:
    """
    Download adjusted close prices and compute returns.

    Returns
    -------
    pd.DataFrame with column 'stock_return'
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No price data found for '{ticker}' between {start} and {end}.")

    prices = raw["Close"].squeeze()

    if frequency == "monthly":
        prices = prices.resample("ME").last()

    returns = prices.pct_change().dropna()
    returns.index = returns.index.tz_localize(None)  # strip timezone for alignment

    df = returns.to_frame(name="stock_return")
    return df


def fetch_ff_factors(
    start: str,
    end: str,
    frequency: Literal["daily", "monthly"] = "daily",
) -> pd.DataFrame:
    """
    Download Fama-French 5-factor data from Ken French's library.
    Returns all 5 factors + RF so any model (CAPM / FF3 / FF5) can slice what it needs.

    Returns
    -------
    pd.DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, RF  (as decimals, not %)
    """
    dataset = _FF_DATASETS[frequency]

    try:
        raw = web.DataReader(dataset, "famafrench", start=start, end=end)[0]
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch Fama-French factors. Check your internet connection.\n"
            f"Original error: {e}"
        )

    # Ken French delivers factors in percent — convert to decimals
    df = raw[_FF5_COLS] / 100.0
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)

    return df
