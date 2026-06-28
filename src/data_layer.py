"""
data_layer.py - load one-CSV-per-stock OHLCV files into aligned panels.

A "panel" = a table where rows are dates, columns are stock tickers.
We build two: one for close prices, one for volume.
Everything downstream (signals, backtest) expects this shape.
"""

import os
import glob
import pandas as pd


def load_one_stock(path: str) -> pd.DataFrame:
    """
    Read a single stock CSV. Expects columns like:
    Date, Open, High, Low, Close, Volume   (case-insensitive)
    Returns a DataFrame indexed by date.
    """
    df = pd.read_csv(path)
    # normalise column names to lowercase so we don't care about Close vs close
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def load_panels(folder: str):
    """
    Read every CSV in `folder`. Ticker = filename without extension.
    Returns (close_panel, volume_panel), both aligned on a shared date index.
    """
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {folder}")

    closes, volumes = {}, {}
    for f in files:
        ticker = os.path.splitext(os.path.basename(f))[0]
        df = load_one_stock(f)
        closes[ticker] = df["close"]
        volumes[ticker] = df["volume"]

    # pd.DataFrame on a dict of Series auto-aligns on the date index
    close_panel = pd.DataFrame(closes).sort_index()
    volume_panel = pd.DataFrame(volumes).sort_index()
    return close_panel, volume_panel


def basic_health_check(close: pd.DataFrame) -> None:
    """Print quick sanity facts - the Phase 0 data audit, in miniature."""
    print(f"  stocks      : {close.shape[1]}")
    print(f"  trading days: {close.shape[0]}")
    print(f"  date range  : {close.index.min().date()} -> {close.index.max().date()}")
    missing = close.isna().sum().sum()
    print(f"  missing vals: {missing}")
    # a crude split detector: any single-day move bigger than 40%?
    daily_ret = close.pct_change()
    cliffs = (daily_ret.abs() > 0.40).sum().sum()
    flag = "  <-- check for unadjusted splits!" if cliffs else ""
    print(f"  >40% 1-day moves: {cliffs}{flag}")
