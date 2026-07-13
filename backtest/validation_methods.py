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
# Every mode applies the SAME two-stage gate, fitted on formation data only:
#   1. pair_is_valid  - correlation floor + positive beta. Runs FIRST, because
#                       hedge_ratio() fits a beta to uncorrelated noise without
#                       complaint, and adf_pvalue() then scores the resulting
#                       nonsense spread without complaint either.
#   2. adf_pvalue     - is that spread actually mean-reverting?
# create_pairs.all_pairs() does no screening at all -- it just enumerates
# within-sector combinations -- so this is the only place pairs get filtered.
#
# Each returns (sharpe, net_pnl, equity_series, margin) or None if the pair is not
# tradeable (not enough data, failed the gate, or not cointegrated in the fit window).

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from candidate_pairs.cointegration import (
    hedge_ratio, spread_with_beta, cointegration_pvalue, pair_is_valid,
    COINT_THRESHOLD,
)
from backtest.zscore_signal import zscore, positions
from backtest.engine import backtest_pair
from backtest.evaluate import sharpe
# single source of truth -- see config.py. FORMATION_MONTHS/STEP_MONTHS are the
# DEFAULTS for run_rolling; callers can override them per-call.
from backtest.config import COST_PER_UNIT, FORMATION_MONTHS, STEP_MONTHS

SPLIT_DATE = "2025-01-01"      # formation/trading boundary for run_split
ZSCORE_WINDOW = 60             # rolling lookback for the z-score, in trading days
MIN_FORMATION_DAYS = 100       # too few days and the fit is not worth trusting
MIN_TRADING_DAYS = 65          # roughly one quarter


def _fit(formation: pd.DataFrame, a: str, b: str):
    """Fit and gate a pair on ONE window. Shared by all three modes so the
    tradeability rules cannot drift apart between them.

    @return (beta, form_spread) if the pair passes, else (None, None).
    """
    if len(formation) < MIN_FORMATION_DAYS:
        return None, None

    beta = hedge_ratio(formation[a], formation[b])

    # gate BEFORE the ADF test -- a beta fitted to noise produces a spread that
    # ADF will still score, sometimes favourably.
    ok, _reason = pair_is_valid(formation[a], formation[b], beta)
    if not ok:
        return None, None

    form_spread = spread_with_beta(formation[a], formation[b], beta)
    if len(form_spread) < MIN_FORMATION_DAYS:
        return None, None

    # Engle-Granger with Phillips-Ouliaris critical values, NOT adfuller() on the
    # fitted residual. beta was estimated by minimizing this very spread's variance,
    # so adfuller() -- which assumes no such search happened -- reports p-values
    # roughly 2x too small. On the first 12 months that difference was 29 pairs
    # passing versus 9, against ~8 expected from chance alone.
    if cointegration_pvalue(formation[a], formation[b], beta) >= COINT_THRESHOLD:
        return None, None

    return beta, form_spread


def _metrics(result: pd.DataFrame, price_a: pd.Series, price_b: pd.Series, beta: float):
    """Sharpe + net P&L + equity + margin from a backtest result.

    margin is the capital actually tied up. The position is 1 unit of A against
    beta units of B, so the combined notional is mean(A) + beta*mean(B) -- NOT
    2*mean(A), which silently assumed the two legs were the same size. Futures
    margin is ~20% of notional.

    margin is returned (not just used internally) so callers can size a multi-pair
    portfolio's total capital -- e.g. to compute % returns for a combined book.
    """
    notional = price_a.mean() + abs(beta) * price_b.mean()
    margin = notional * 0.20
    return sharpe(result["net_pnl"] / margin), result["net_pnl"].sum(), result["equity"], margin


def run_full(close: pd.DataFrame, a: str, b: str):
    """In-sample baseline. Fits and trades on the same data -- diagnostic only."""
    pair = close[[a, b]].dropna()
    beta, _ = _fit(pair, a, b)
    if beta is None:
        return None

    s = spread_with_beta(pair[a], pair[b], beta)
    result = backtest_pair(s, positions(zscore(s, window=ZSCORE_WINDOW)),
                           cost_per_unit=COST_PER_UNIT)
    return _metrics(result, pair[a], pair[b], beta)


def run_split(close: pd.DataFrame, a: str, b: str):
    """One honest out-of-sample test: fit before SPLIT_DATE, trade after it."""
    formation = close[close.index < SPLIT_DATE][[a, b]].dropna()
    trading = close[close.index >= SPLIT_DATE][[a, b]].dropna()

    beta, _ = _fit(formation, a, b)
    if beta is None:
        return None

    trade_spread = spread_with_beta(trading[a], trading[b], beta)
    if len(trade_spread) < MIN_TRADING_DAYS:
        return None

    result = backtest_pair(trade_spread,
                           positions(zscore(trade_spread, window=ZSCORE_WINDOW)),
                           cost_per_unit=COST_PER_UNIT)
    return _metrics(result, trading[a], trading[b], beta)


def run_rolling(close: pd.DataFrame, a: str, b: str,
                formation_months: int = FORMATION_MONTHS,
                step_months: int = STEP_MONTHS):
    """Walk forward: re-fit on the trailing `formation_months`, trade the next
    `step_months`, repeat. Window shape is a parameter so the same logic covers
    e.g. 24/3 and 12/3 without a duplicated 'three_month_rolling' method.

    A pair that fails the gate in one window is simply not traded that window,
    and can re-qualify later -- which is what a live desk would do.

    @param formation_months  trailing months used to gate the pair + fit beta
    @param step_months       months traded forward per fitted beta before re-fitting
    """
    pair = close[[a, b]].dropna()
    if pair.empty:
        return None
    start, end = pair.index.min(), pair.index.max()
    split = start + pd.DateOffset(months=formation_months)

    window_results = []
    window_betas = []          # one beta per window that actually traded
    while split < end:
        trade_end = split + pd.DateOffset(months=step_months)
        # formation is the TRAILING window only, not all history before `split`,
        # so shortening formation_months genuinely shortens the fitting sample.
        form_start = split - pd.DateOffset(months=formation_months)
        formation = pair[(pair.index >= form_start) & (pair.index < split)]
        trading = pair[(pair.index >= split) & (pair.index < trade_end)]
        window_start = split
        split = trade_end

        if len(trading) < 5:
            continue

        beta, form_spread = _fit(formation, a, b)
        if beta is None:
            continue          # pair failed the gate this window -- sit it out
        window_betas.append(beta)

        # Build the spread over formation+trading together (same frozen beta) so
        # the rolling z-score window arrives at day 1 of trading already calibrated
        # from formation history, instead of re-warming from scratch every
        # step_months. With step_months=3 (~63 trading days) and a 60-day z-score
        # window, computing z-score fresh on trade_spread alone left only ~3
        # tradeable days per quarter -- the rest was NaN warm-up, discarded at
        # every window boundary.
        combined = pair[(pair.index >= form_start) & (pair.index < trade_end)]
        combined_spread = spread_with_beta(combined[a], combined[b], beta)
        z_combined = zscore(combined_spread,
                            window=min(ZSCORE_WINDOW, len(form_spread) - 1))

        trade_spread = combined_spread[combined_spread.index >= window_start]
        z = z_combined[z_combined.index >= window_start]
        window_results.append(backtest_pair(trade_spread, positions(z),
                                            cost_per_unit=COST_PER_UNIT))

    if not window_results:
        return None
    stitched = pd.concat(window_results)
    stitched["equity"] = stitched["net_pnl"].cumsum()

    # Margin uses the MEAN of the betas that actually traded.
    #
    # This used to read `beta` -- the loop variable. That was a latent bug that the
    # gate exposed: _fit() returns (None, None) for a window that fails the gate,
    # which rebinds `beta` to None. If the FINAL window was rejected, `beta` was
    # None when the loop exited, and _metrics blew up on abs(None). Before the gate
    # existed, beta was never rebound to None, so it never surfaced.
    #
    # Averaging is also more honest than taking the last one: beta is re-fitted every
    # window, so no single value describes the capital tied up across the whole run.
    avg_beta = float(np.mean(window_betas))
    return _metrics(stitched, pair[a], pair[b], avg_beta)


# Run file to test, returns results with run_full, run_split and run_rolling
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
