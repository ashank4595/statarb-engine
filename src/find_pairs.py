"""
find_pairs.py - Steps 3-6 of pair discovery: the statistical screen.

Takes the economically-motivated candidate pairs (from sectors.py) and runs
them through real statistics to decide which ones are GENUINELY tradeable.
NO trading, NO P&L here - this is pure screening. The output is a ranked
shortlist of pairs worth backtesting later.

Pipeline per candidate pair:
  1. correlation pre-screen  (cheap filter, drop obvious non-movers)
  2. cointegration test      (Engle-Granger + ADF on the spread - the real test)
  3. half-life of reversion  (how many days to revert - too slow/fast = skip)

Requires a `close` price panel (dates x tickers). Wire up data_layer first.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


def hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """Regress A on B; the slope is how many units of B hedge one unit of A."""
    b_const = sm.add_constant(price_b)
    model = sm.OLS(price_a, b_const).fit()
    return model.params.iloc[1]


def spread_series(price_a: pd.Series, price_b: pd.Series) -> pd.Series:
    """The tradeable quantity: A - beta*B. If cointegrated, this mean-reverts."""
    beta = hedge_ratio(price_a, price_b)
    return price_a - beta * price_b


def half_life(spread: pd.Series) -> float:
    """
    Estimate days for the spread to revert halfway to its mean.
    Models spread as Ornstein-Uhlenbeck: regress dSpread on lagged spread.
    """
    lag = spread.shift(1).dropna()
    delta = (spread - spread.shift(1)).dropna()
    lag = lag.loc[delta.index]
    beta = sm.OLS(delta, sm.add_constant(lag)).fit().params.iloc[1]
    if beta >= 0:
        return np.inf  # not mean-reverting
    return -np.log(2) / beta


def screen_pair(price_a: pd.Series, price_b: pd.Series) -> dict:
    """Run the full statistical screen on one pair. Returns a row of metrics."""
    joined = pd.concat([price_a, price_b], axis=1).dropna()
    a, b = joined.iloc[:, 0], joined.iloc[:, 1]

    corr = a.pct_change().corr(b.pct_change())
    # Engle-Granger cointegration test - low p-value = cointegrated (good)
    _, coint_p, _ = coint(a, b)
    spread = spread_series(a, b)
    # ADF on the spread - low p-value = spread is stationary/mean-reverting
    adf_p = adfuller(spread.dropna())[1]
    hl = half_life(spread)

    return {
        "correlation": round(corr, 3),
        "coint_pvalue": round(coint_p, 4),
        "adf_pvalue": round(adf_p, 4),
        "half_life_days": round(hl, 1) if np.isfinite(hl) else None,
    }


def screen_all(close: pd.DataFrame, candidate_pairs: list,
               coint_threshold: float = 0.05) -> pd.DataFrame:
    """
    Screen every candidate pair. Returns a ranked table, best first.
    A pair 'passes' if its cointegration p-value is below the threshold.
    """
    rows = []
    for sector, a, b in candidate_pairs:
        if a not in close.columns or b not in close.columns:
            continue
        metrics = screen_pair(close[a], close[b])
        metrics.update({"sector": sector, "stock_a": a, "stock_b": b})
        metrics["passes"] = metrics["coint_pvalue"] < coint_threshold
        rows.append(metrics)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cols = ["sector", "stock_a", "stock_b", "correlation",
            "coint_pvalue", "adf_pvalue", "half_life_days", "passes"]
    return df[cols].sort_values("coint_pvalue").reset_index(drop=True)
