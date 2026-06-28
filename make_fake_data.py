"""
make_fake_data.py - generates synthetic per-stock OHLCV CSVs in the SAME
format as your real files (Date,Open,High,Low,Close,Volume), then runs the
loader so you can see things working before touching real data.
"""

import os
import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from data_layer import load_panels, basic_health_check

RAW = os.path.join(os.path.dirname(__file__), "data", "raw")
os.makedirs(RAW, exist_ok=True)

rng = np.random.default_rng(0)
dates = pd.bdate_range("2021-01-01", periods=1000)  # ~4 yrs of business days
tickers = ["HDFCBANK", "ICICIBANK", "INFY", "TCS", "RELIANCE"]

for t in tickers:
    # random-walk close price starting near 1000
    rets = rng.normal(0.0003, 0.015, len(dates))
    close = 1000 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.003, len(dates)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, len(dates))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, len(dates))))
    vol = rng.integers(1_000_000, 5_000_000, len(dates))
    df = pd.DataFrame(
        {"Date": dates, "Open": open_, "High": high,
         "Low": low, "Close": close, "Volume": vol}
    )
    df.to_csv(os.path.join(RAW, f"{t}.csv"), index=False)

print(f"Wrote {len(tickers)} fake CSVs to data/raw/\n")

close_panel, volume_panel = load_panels(RAW)
print("Loaded close panel - first 3 rows:")
print(close_panel.head(3).round(1))
print("\nHealth check:")
basic_health_check(close_panel)
