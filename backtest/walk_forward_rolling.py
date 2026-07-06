# walk_forward_rolling.py
# PURPOSE: full walk-forward validation for ONE pair (version B).
#
# The single split (walk_forward.py) learned beta ONCE on 2023-2024 and used
# that stale beta for all of 2025-2026 - it could not adapt when the pair's
# relationship drifted. This rolling version re-fits periodically:
#
#   every STEP months, re-learn beta and re-check cointegration using ALL data
#   up to that point (never ahead), then trade only the next STEP months.
#
# Two advantages over the single split:
#   1. beta stays current (adapts as the relationship drifts)
#   2. if a pair STOPS passing cointegration, we skip trading it that window
#      (drops dead pairs instead of riding them down - what real desks do)
#
# The stitched trading-window P&Ls form one continuous out-of-sample equity
# curve. No look-ahead: each window fits on past data, trades the next window.
# Run: python3 -m backtest.walk_forward_rolling

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt

from data_layer import load_panel, load_daily_close
from candidate_pairs.cointegration import hedge_ratio, spread, spread_with_beta, adf_pvalue
from backtest.zscore_signal import zscore, positions
from backtest.engine import backtest_pair
from backtest.evaluate import summary

FOLDER = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"

# rolling parameters
FORMATION_MONTHS = 24      # how much history to fit on each window
STEP_MONTHS      = 3       # how far forward to trade before re-fitting
COINT_THRESHOLD  = 0.05    # pair must still pass this to be traded that window

# the pair to validate
STOCK_A = "HAVELLS"
STOCK_B = "CROMPTON"


def run_rolling(stock_a: str, stock_b: str):
    close = load_panel(FOLDER, tickers=[stock_a, stock_b])
    close = close.dropna()

    start = close.index.min()
    end = close.index.max()

    # first trading window begins after the initial formation period
    split = start + pd.DateOffset(months=FORMATION_MONTHS)

    window_results = []   # collect each trading window's P&L result
    skipped = 0
    traded = 0

    while split < end:
        trade_end = split + pd.DateOffset(months=STEP_MONTHS)

        # formation = everything from start up to (not including) the split point
        formation = close[close.index < split]
        # trading = the next STEP months (never ahead of what we know)
        trading = close[(close.index >= split) & (close.index < trade_end)]

        if len(formation) < 100 or len(trading) < 5:
            split = trade_end
            continue

        # re-check cointegration on formation only
        form_spread = spread(formation[stock_a], formation[stock_b])
        if len(form_spread) < 100:
            split = trade_end
            continue
        p = adf_pvalue(form_spread)

        if p >= COINT_THRESHOLD:
            # pair stopped passing -> skip trading it this window
            skipped += 1
            split = trade_end
            continue

        # pair still passes: learn beta on formation, apply to trading
        beta = hedge_ratio(formation[stock_a], formation[stock_b])
        trade_spread = spread_with_beta(trading[stock_a], trading[stock_b], beta)

        z = zscore(trade_spread, window=min(60, len(trade_spread) - 1))
        pos = positions(z)
        result = backtest_pair(trade_spread, pos)
        window_results.append(result)
        traded += 1

        split = trade_end

    if not window_results:
        print("No tradeable windows.")
        return

    # stitch all trading-window P&Ls into one continuous series
    stitched = pd.concat(window_results)
    stitched["equity"] = stitched["net_pnl"].cumsum()

    margin = close[stock_a].mean() * 2 * 0.20
    nifty = load_daily_close(os.path.join(FOLDER, "NIFTY_-I.csv"))
    nifty_returns = nifty.pct_change()

    print(f"=== {stock_a}-{stock_b} ROLLING WALK-FORWARD ===")
    print(f"windows traded : {traded}")
    print(f"windows skipped: {skipped}  (pair failed cointegration re-check)")
    print()
    summary(stitched, margin, index_returns=nifty_returns)

    stitched["equity"].plot(
        title=f"{stock_a}-{stock_b} rolling walk-forward equity",
        figsize=(12, 5))
    plt.axhline(0, color="black", alpha=0.3)
    plt.show()


if __name__ == "__main__":
    run_rolling(STOCK_A, STOCK_B)
