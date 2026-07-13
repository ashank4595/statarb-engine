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
#   run_rolling(close, a, b, formation_months=..., step_months=...)
#       Re-fit beta and re-check cointegration every step_months using only the
#       trailing formation_months of data, trading the next step_months forward
#       each time and stitching the P&L. Most realistic: beta stays current and
#       dead pairs get skipped. The window shape is a parameter, not a new mode --
#       a 12-month formation traded 3 months forward is just
#       run_rolling(..., formation_months=12, step_months=3). Defaults come from
#       config.FORMATION_MONTHS / config.STEP_MONTHS.
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
# single source of truth -- see config.py. FORMATION_MONTHS/STEP_MONTHS are the
# DEFAULTS for run_rolling; callers can override them per-call.
from backtest.config import COST_PER_UNIT, FORMATION_MONTHS, STEP_MONTHS

SPLIT_DATE = "2025-01-01"      # formation/trading boundary for run_split
COINT_THRESHOLD = 0.05         # a pair must pass ADF below this to be traded
ZSCORE_WINDOW = 60             # rolling lookback for the z-score, in trading days


def _metrics(result: pd.DataFrame, ref_price: pd.Series):
    """Sharpe + net P&L + equity + margin from a backtest result, given a price for margin.
    margin is returned (not just used internally) so callers can size a multi-pair
    portfolio's total capital -- e.g. to compute % returns for a combined book."""
    margin = ref_price.mean() * 2 * 0.20
    return sharpe(result["net_pnl"] / margin), result["net_pnl"].sum(), result["equity"], margin


def run_full(close: pd.DataFrame, a: str, b: str):
    s = spread(close[a], close[b])
    if len(s) < 100 or adf_pvalue(s) >= COINT_THRESHOLD:
        return None
    result = backtest_pair(s, positions(zscore(s)), cost_per_unit=COST_PER_UNIT)
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
    result = backtest_pair(trade_spread, positions(zscore(trade_spread)), cost_per_unit=COST_PER_UNIT)
    return _metrics(result, trading[a])


def run_rolling(close: pd.DataFrame, a: str, b: str,
                formation_months: int = FORMATION_MONTHS,
                step_months: int = STEP_MONTHS):
    """Walk forward: re-fit on the trailing `formation_months`, trade the next
    `step_months`, repeat. Window shape is a parameter so the same logic covers
    e.g. 24/3 and 12/3 without a duplicated 'three_month_rolling' method.

    @param formation_months  trailing months used to test cointegration + fit beta
    @param step_months       months traded forward per fitted beta before re-fitting
    """
    pair = close[[a, b]].dropna()
    if pair.empty:
        return None
    start, end = pair.index.min(), pair.index.max()
    split = start + pd.DateOffset(months=formation_months)

    window_results = []
    while split < end:
        trade_end = split + pd.DateOffset(months=step_months)
        # formation is the TRAILING window only, not all history before `split`,
        # so shortening formation_months genuinely shortens the fitting sample.
        form_start = split - pd.DateOffset(months=formation_months)
        formation = pair[(pair.index >= form_start) & (pair.index < split)]
        trading = pair[(pair.index >= split) & (pair.index < trade_end)]
        split = trade_end

        if len(formation) < 100 or len(trading) < 5:
            continue
        form_spread = spread(formation[a], formation[b])
        if len(form_spread) < 100 or adf_pvalue(form_spread) >= COINT_THRESHOLD:
            continue
        beta = hedge_ratio(formation[a].dropna(), formation[b].dropna())

        # Build the spread over formation+trading together (same frozen beta) so
        # the rolling z-score window arrives at day 1 of trading already calibrated
        # from formation history, instead of re-warming from scratch every
        # step_months. With step_months=3 (~63 trading days) and a 60-day z-score
        # window, computing z-score fresh on trade_spread alone left only ~3
        # tradeable days per quarter -- the rest was NaN warm-up, discarded at
        # every window boundary.
        combined = pair[(pair.index >= form_start) & (pair.index < trade_end)]
        combined_spread = spread_with_beta(combined[a], combined[b], beta)
        z_combined = zscore(combined_spread, window=min(ZSCORE_WINDOW, len(form_spread) - 1))

        trade_spread = combined_spread[combined_spread.index >= split]
        z = z_combined[z_combined.index >= split]
        window_results.append(backtest_pair(trade_spread, positions(z), cost_per_unit=COST_PER_UNIT))

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
        sh, pnl, equity, margin = out
        print(f"{name:8s}  sharpe={sh:.2f}  net_pnl={pnl:.1f}  margin={margin:.1f}")
        equity.plot(ax=ax, title=f"{name}  (Sharpe {sh:.2f})")
        ax.axhline(0, color="black", alpha=0.3)

    plt.tight_layout()
    plt.show()
