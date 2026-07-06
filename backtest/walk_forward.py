# walk_forward.py
# PURPOSE: out-of-sample validation via a single formation/trading split.
#
# All metrics so far are IN-SAMPLE: a pair was selected because it looked
# cointegrated over the whole period, then measured over that same period.
# That is circular. This runner separates the two:
#
#   FORMATION (2023-2024): select cointegrated pairs, learn each beta.
#   TRADING   (2025-2026): trade those exact pairs with the formation beta,
#                          on data never used for selection or fitting.
#
# The gap between formation-period and trading-period performance is the
# overfitting measure. A real edge holds up out-of-sample; in-sample luck
# collapses. Run: python3 -m backtest.walk_forward

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt

from data_layer import load_panel, load_daily_close
from candidate_pairs.cointegration import (
    hedge_ratio, spread, spread_with_beta, adf_pvalue, half_life,
)
from backtest.zscore_signal import zscore, positions
from backtest.engine import backtest_pair
from backtest.evaluate import summary

FOLDER = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"
SPLIT_DATE = "2025-01-01"          # everything before = formation, after = trading

# the pair to validate (swap to test others)
STOCK_A = "HAVELLS"
STOCK_B = "CROMPTON"


def run_split(stock_a: str, stock_b: str):
    close = load_panel(FOLDER, tickers=[stock_a, stock_b])

    # split by date
    formation = close[close.index < SPLIT_DATE]
    trading   = close[close.index >= SPLIT_DATE]

    # --- FORMATION: learn beta, check the pair is cointegrated here ---
    beta = hedge_ratio(formation[stock_a].dropna(), formation[stock_b].dropna())
    form_spread = spread(formation[stock_a], formation[stock_b])
    form_p = adf_pvalue(form_spread)
    print(f"formation beta        : {beta:.3f}")
    print(f"formation ADF p-value : {form_p:.4f}  (<0.05 = cointegrated in-sample)")

    # --- TRADING: apply the SAME beta out-of-sample ---
    trade_spread = spread_with_beta(trading[stock_a], trading[stock_b], beta)
    z = zscore(trade_spread)
    p = positions(z)
    result = backtest_pair(trade_spread, p)

    # margin + Nifty for the metrics
    margin = trading[stock_a].mean() * 2 * 0.20
    nifty = load_daily_close(os.path.join(FOLDER, "NIFTY_-I.csv"))
    nifty_returns = nifty.pct_change()

    print(f"\n=== {stock_a}-{stock_b} OUT-OF-SAMPLE (trading period) ===")
    summary(result, margin, index_returns=nifty_returns)

    result["equity"].plot(
        title=f"{stock_a}-{stock_b} out-of-sample equity ({SPLIT_DATE} onward)",
        figsize=(12, 5))
    plt.axhline(0, color="black", alpha=0.3)
    plt.show()


if __name__ == "__main__":
    run_split(STOCK_A, STOCK_B)
