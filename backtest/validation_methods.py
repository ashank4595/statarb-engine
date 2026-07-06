# validation_methods.py
# Three ways to backtest one pair, ordered from optimistic to honest:
#
#   run_full(close, a, b)
#       Fit beta and trade on ALL data. In-sample, so it flatters the result -
#       the pair was chosen using the same data it is scored on. Baseline only.
#
#   run_split(close, a, b)
#       Fit beta + check cointegration on the formation period (before SPLIT_DATE),
#       then trade the later period once with that fixed beta. First honest test:
#       the trading period never influenced pair selection or beta.
#
#   run_rolling(close, a, b)
#       Re-fit beta and re-check cointegration every STEP_MONTHS using only past
#       data, trading the next STEP_MONTHS forward each time and stitching the P&L.
#       Most realistic: beta stays current and dead pairs get skipped.
#
# Each returns (sharpe, net_pnl, equity_series) or None if the pair is not
# tradeable (not enough data, or not cointegrated in the fitting period).

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from candidate_pairs.cointegration import hedge_ratio, spread, spread_with_beta, adf_pvalue
from backtest.zscore_signal import zscore, positions
from backtest.engine import backtest_pair
from backtest.evaluate import sharpe

SPLIT_DATE = "2025-01-01"      # formation/trading boundary for run_split
FORMATION_MONTHS = 24          # formation length for run_rolling
STEP_MONTHS = 3                # trade-forward step for run_rolling
COINT_THRESHOLD = 0.05         # a pair must pass ADF below this to be traded


def _metrics(result: pd.DataFrame, ref_price: pd.Series):
    """Sharpe + net P&L + equity from a backtest result, given a price for margin."""
    margin = ref_price.mean() * 2 * 0.20
    return sharpe(result["net_pnl"] / margin), result["net_pnl"].sum(), result["equity"]


def run_full(close: pd.DataFrame, a: str, b: str):
    s = spread(close[a], close[b])
    if len(s) < 100 or adf_pvalue(s) >= COINT_THRESHOLD:
        return None
    result = backtest_pair(s, positions(zscore(s)))
    return _metrics(result, close[a])


def run_split(close: pd.DataFrame, a: str, b: str):
    formation = close[close.index < SPLIT_DATE]
    trading = close[close.index >= SPLIT_DATE]

    form_spread = spread(formation[a], formation[b])
    if len(form_spread) < 100 or adf_pvalue(form_spread) >= COINT_THRESHOLD:
        return None

    beta = hedge_ratio(formation[a].dropna(), formation[b].dropna())
    trade_spread = spread_with_beta(trading[a], trading[b], beta)
    if len(trade_spread) < 65:
        return None
    result = backtest_pair(trade_spread, positions(zscore(trade_spread)))
    return _metrics(result, trading[a])


def run_rolling(close: pd.DataFrame, a: str, b: str):
    pair = close[[a, b]].dropna()
    if pair.empty:
        return None
    start, end = pair.index.min(), pair.index.max()
    split = start + pd.DateOffset(months=FORMATION_MONTHS)

    window_results = []
    while split < end:
        trade_end = split + pd.DateOffset(months=STEP_MONTHS)
        formation = pair[pair.index < split]
        trading = pair[(pair.index >= split) & (pair.index < trade_end)]
        split = trade_end

        if len(formation) < 100 or len(trading) < 5:
            continue
        form_spread = spread(formation[a], formation[b])
        if len(form_spread) < 100 or adf_pvalue(form_spread) >= COINT_THRESHOLD:
            continue
        beta = hedge_ratio(formation[a].dropna(), formation[b].dropna())
        trade_spread = spread_with_beta(trading[a], trading[b], beta)
        z = zscore(trade_spread, window=min(60, len(trade_spread) - 1))
        window_results.append(backtest_pair(trade_spread, positions(z)))

    if not window_results:
        return None
    stitched = pd.concat(window_results)
    stitched["equity"] = stitched["net_pnl"].cumsum()
    return _metrics(stitched, pair[a])

# Run file to test, returns results with run_full, run_split and run rolling
if __name__ == "__main__":
    # inspect one pair under all three methods, plot the equity curves
    from data_layer import load_panel
    import matplotlib.pyplot as plt

    FOLDER = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"
    STOCK_A, STOCK_B = "ONGC", "COALINDIA"

    close = load_panel(FOLDER, tickers=[STOCK_A, STOCK_B])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (name, fn) in zip(axes, [("full", run_full),
                                      ("split", run_split),
                                      ("rolling", run_rolling)]):
        out = fn(close, STOCK_A, STOCK_B)
        if out is None:
            ax.set_title(f"{name}: not tradeable")
            continue
        sh, pnl, equity = out
        print(f"{name:8s}  sharpe={sh:.2f}  net_pnl={pnl:.1f}")
        equity.plot(ax=ax, title=f"{name}  (Sharpe {sh:.2f})")
        ax.axhline(0, color="black", alpha=0.3)

    plt.tight_layout()
    plt.show()
