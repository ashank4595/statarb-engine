"""
data_layer.py - load NSE near-month futures (1-min bars) into a daily price panel.

Your raw files (derived_data/futures/TICKER_-I.csv) look like:
    Ticker,Date,Time,Open,High,Low,Close,Volume,Open Interest
    HDFCBANK-I.NFO,02/01/2023,09:15:59,1630.05,...

This loader:
  - reads only near-month (_-I.csv) files
  - parses the dd/mm/yyyy dates
  - compresses 1-minute bars down to ONE row per day (daily close = last
    price of the day; daily volume = sum) - much lighter to work with
  - builds a `close` panel: rows = dates, cols = tickers

Daily is the clean starting point. We can switch to finer bars later without
changing anything downstream.
"""

import os
import glob
import pandas as pd


def _clean_ticker(filename: str) -> str:
    """'HDFCBANK_-I.csv' -> 'HDFCBANK'."""
    base = os.path.basename(filename)
    base = base.replace(".csv", "")
    base = base.replace("_-I", "")   # strip the near-month suffix
    return base


def load_daily_close(path: str) -> pd.Series:
    """
    Read one near-month futures CSV of 1-min bars, return a DAILY close series.
    Daily close = the last traded price of each day.
    """
    df = pd.read_csv(path, usecols=["Date", "Time", "Close", "Volume"])
    # dates are dd/mm/yyyy
    df["date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    # last close of each day = end-of-day price
    daily = df.groupby("date")["Close"].last()
    return daily


def load_panel(folder: str, only_near_month: bool = True,
               tickers: list | None = None) -> pd.DataFrame:
    """
    Build a daily close panel from all near-month futures files in `folder`.

    tickers: optional whitelist (e.g. only the ones in your sectors.py). If
             given, we only load those - much faster than loading all ~220.
    Returns DataFrame: rows = dates, cols = tickers, values = daily close.
    """
    pattern = "*_-I.csv" if only_near_month else "*.csv"
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")

    closes = {}
    for f in files:
        ticker = _clean_ticker(f)
        if tickers is not None and ticker not in tickers:
            continue
        try:
            closes[ticker] = load_daily_close(f)
        except Exception as e:
            print(f"  skipped {ticker}: {e}")

    panel = pd.DataFrame(closes).sort_index()
    return panel


def health_check(close: pd.DataFrame) -> None:
    print(f"  stocks      : {close.shape[1]}")
    print(f"  trading days: {close.shape[0]}")
    print(f"  date range  : {close.index.min().date()} -> {close.index.max().date()}")
    print(f"  missing vals: {close.isna().sum().sum()}")
    daily_ret = close.pct_change()
    cliffs = (daily_ret.abs() > 0.40).sum().sum()
    flag = "  <-- likely stock splits (corporate actions) to handle later" if cliffs else ""
    print(f"  >40% 1-day moves: {cliffs}{flag}")
