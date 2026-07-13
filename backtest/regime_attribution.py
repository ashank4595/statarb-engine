# regime_attribution.py
# Answers: what fraction of the portfolio's drawdown occurred in low-trend,
# whipsaw-prone regimes?  Produces the [XX]% for the resume line.
#
# Method (descriptive attribution, not a trading signal -- regime labels use
# full trading-period percentiles, which is fine because nothing here feeds
# back into positions):
#   1. Rebuild each tradeable pair's TRADING-period spread exactly as the
#      current MODE does (frozen formation beta for split/rolling), so the
#      regime is measured on the same series the strategy actually traded.
#   2. Kaufman efficiency ratio per spread:
#          ER_t = |s_t - s_{t-W}| / sum_{i=t-W+1..t} |s_i - s_{i-1}|
#      ER near 1 = clean directional move; ER near 0 = churn/whipsaw.
#      Numeric example (W=4): spread path 100,101,100,101,100 -> net move 0,
#      path length 4, ER = 0/4 = 0 (pure whipsaw). Path 100,101,102,103,104
#      -> ER = 4/4 = 1 (clean trend).
#   3. Percentile-rank each pair's ER over its own history (scale-free across
#      pairs), then combine into one daily portfolio "chop score" using the
#      same equal-risk weights portfolio.py uses.
#   4. Label the bottom WHIPSAW_QUANTILE of chop-score days "whipsaw", the
#      rest "normal".
#   5. Attribute drawdown by DEEPENING: with dd_t = equity_t - cummax(equity),
#      deepening_t = max(0, -(dd_t - dd_{t-1})) is the new drawdown created on
#      day t. Regime shares of total deepening sum to exactly 100%.
#      Baseline: if drawdown were regime-independent, the whipsaw share would
#      equal its day share (~ WHIPSAW_QUANTILE). XX% above that = concentration.
#
# Run: python3 -m backtest.regime_attribution
# Writes: results/regime_attribution_{MODE}.csv (per-day score/regime/deepening)

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data_layer import load_panel
from candidate_pairs.create_pairs import all_pairs
from candidate_pairs.cointegration import hedge_ratio, spread, spread_with_beta, adf_pvalue
from backtest.portfolio import combine
from backtest.config import MODE, WEIGHT_SCHEME
from backtest.validation_methods import (
    SPLIT_DATE, FORMATION_MONTHS, STEP_MONTHS, COINT_THRESHOLD,
)

FOLDER = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"

ER_WINDOW = 20          # days for the efficiency ratio (~1 trading month)
WHIPSAW_QUANTILE = 1/3  # bottom third of chop-score days = "whipsaw" regime


def efficiency_ratio(spread_series: pd.Series, window: int = ER_WINDOW) -> pd.Series:
    """
    Kaufman efficiency ratio of a spread over a rolling window.

    :param spread_series: daily spread values (a - beta*b)
    :param window: lookback in trading days
    :return: series in [0, 1]; first `window` values are NaN (warm-up)
    """
    net_move = (spread_series - spread_series.shift(window)).abs()
    path_length = spread_series.diff().abs().rolling(window).sum()
    return net_move / path_length.replace(0, np.nan)


def _trade_spread_full(close: pd.DataFrame, a: str, b: str) -> pd.Series | None:
    """Spread over ALL data, fresh beta -- mirrors run_full."""
    s = spread(close[a], close[b])
    if len(s) < 100 or adf_pvalue(s) >= COINT_THRESHOLD:
        return None
    return s


def _trade_spread_split(close: pd.DataFrame, a: str, b: str) -> pd.Series | None:
    """Trading-period spread with frozen formation beta -- mirrors run_split."""
    formation = close[close.index < SPLIT_DATE]
    trading = close[close.index >= SPLIT_DATE]

    form_spread = spread(formation[a], formation[b])
    if len(form_spread) < 100 or adf_pvalue(form_spread) >= COINT_THRESHOLD:
        return None
    beta = hedge_ratio(formation[a].dropna(), formation[b].dropna())
    trade_spread = spread_with_beta(trading[a], trading[b], beta)
    if len(trade_spread) < 65:
        return None
    return trade_spread


def _trade_spread_rolling(close: pd.DataFrame, a: str, b: str) -> pd.Series | None:
    """Stitched trading-window spreads, frozen beta per window -- mirrors run_rolling."""
    pair = close[[a, b]].dropna()
    if pair.empty:
        return None
    start, end = pair.index.min(), pair.index.max()
    split = start + pd.DateOffset(months=FORMATION_MONTHS)

    windows = []
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
        combined = pair[pair.index < trade_end]
        combined_spread = spread_with_beta(combined[a], combined[b], beta)
        windows.append(combined_spread[combined_spread.index >= trading.index.min()])

    if not windows:
        return None
    return pd.concat(windows)


_SPREAD_BUILDERS = {
    "full": _trade_spread_full,
    "split": _trade_spread_split,
    "rolling": _trade_spread_rolling,
}


def chop_score(spreads: dict[str, pd.Series], weights: pd.Series,
               window: int = ER_WINDOW) -> pd.Series:
    """
    Daily portfolio chop score: weighted mean of per-pair ER percentile ranks.

    Each pair's ER is percentile-ranked over its own trading-period history so
    pairs with different spread scales contribute comparably. Weights are
    renormalized daily over the pairs that have a value that day.

    :param spreads: pair name -> trading-period spread series
    :param weights: pair name -> portfolio weight (from portfolio.combine)
    :param window: efficiency-ratio lookback
    :return: daily score in [0, 1]; LOW = whipsaw-prone
    """
    ranks = {}
    for name, s in spreads.items():
        if name not in weights.index:
            continue
        er = efficiency_ratio(s, window).dropna()
        if len(er) == 0:
            continue
        ranks[name] = er.rank(pct=True)
    rank_df = pd.concat(ranks, axis=1).sort_index()

    w = weights.reindex(rank_df.columns)
    mask = rank_df.notna()
    daily_w = mask.mul(w, axis=1)
    return (rank_df * daily_w).sum(axis=1) / daily_w.sum(axis=1)


def classify_regimes(score: pd.Series,
                     whipsaw_quantile: float = WHIPSAW_QUANTILE) -> pd.Series:
    """
    Label each day "whipsaw" (bottom quantile of chop score) or "normal".

    :param score: daily chop score from chop_score()
    :param whipsaw_quantile: fraction of days labeled whipsaw
    :return: series of {"whipsaw", "normal"} on score's index
    """
    threshold = score.quantile(whipsaw_quantile)
    return pd.Series(np.where(score <= threshold, "whipsaw", "normal"),
                     index=score.index)


def drawdown_attribution(equity: pd.Series, regimes: pd.Series) -> dict:
    """
    Decompose drawdown deepening by regime.

    dd_t = equity_t - cummax(equity); deepening_t = max(0, -(dd_t - dd_{t-1})).
    Sum of deepening over all days = total new drawdown created; regime shares
    of it sum to 100%. Days without a regime label (ER warm-up) are excluded
    from both numerator and denominator.

    :param equity: portfolio cumulative P&L (currency units)
    :param regimes: daily {"whipsaw","normal"} labels
    :return: dict with regime shares, day shares, max-DD episode breakdown,
             and the aligned per-day frame used (key "daily")
    """
    dd = equity - equity.cummax()
    deepening = (-dd.diff()).clip(lower=0)
    deepening.iloc[0] = -dd.iloc[0]

    df = pd.DataFrame({"equity": equity, "dd": dd, "deepening": deepening})
    df["regime"] = regimes.reindex(df.index)
    n_unlabeled = int(df["regime"].isna().sum())
    labeled = df.dropna(subset=["regime"])

    total = labeled["deepening"].sum()
    shares = labeled.groupby("regime")["deepening"].sum() / total
    day_shares = labeled["regime"].value_counts(normalize=True)

    # max drawdown episode: peak -> trough of the deepest drawdown
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    episode = labeled.loc[peak:trough]
    episode_total = episode["deepening"].sum()
    episode_shares = (episode.groupby("regime")["deepening"].sum() / episode_total
                      if episode_total > 0 else pd.Series(dtype=float))

    return {
        "shares": shares,                  # <- shares["whipsaw"] is the XX%
        "day_shares": day_shares,
        "max_dd": float(dd.min()),
        "max_dd_peak": peak,
        "max_dd_trough": trough,
        "episode_shares": episode_shares,
        "n_unlabeled": n_unlabeled,
        "daily": df,
    }


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(root, "results")

    # portfolio equity + the same weights portfolio.py used
    daily_pnl = pd.read_csv(os.path.join(results_dir, f"daily_pnl_{MODE}.csv"),
                            index_col=0, parse_dates=True)
    summary = pd.read_csv(os.path.join(root, f"results_{MODE}.csv"))
    summary["pair"] = summary["stock_a"] + "-" + summary["stock_b"]
    margins = summary.set_index("pair")["margin"]
    portfolio = combine(daily_pnl, margins, scheme=WEIGHT_SCHEME)
    weights = portfolio.attrs["weights"]

    # rebuild each traded pair's spread under the current MODE
    builder = _SPREAD_BUILDERS[MODE]
    traded = set(daily_pnl.columns)
    pairs = [(a, b) for _, a, b in all_pairs() if f"{a}-{b}" in traded]
    tickers = sorted({t for a, b in pairs for t in [a, b]})
    print(f"loading {len(tickers)} tickers, mode = {MODE}...")
    close = load_panel(FOLDER, tickers=tickers)

    spreads = {}
    for a, b in pairs:
        if a not in close.columns or b not in close.columns:
            continue
        s = builder(close, a, b)
        if s is not None:
            spreads[f"{a}-{b}"] = s
    print(f"rebuilt spreads for {len(spreads)} of {len(traded)} traded pairs")

    score = chop_score(spreads, weights)
    regimes = classify_regimes(score)
    out = drawdown_attribution(portfolio["equity"], regimes)

    xx = 100 * out["shares"].get("whipsaw", 0.0)
    day_share = 100 * out["day_shares"].get("whipsaw", 0.0)
    print("=" * 60)
    print(f"mode / weights          : {MODE} / {WEIGHT_SCHEME}")
    print(f"ER window / whipsaw q   : {ER_WINDOW}d / bottom {WHIPSAW_QUANTILE:.0%}")
    print(f"whipsaw share of days   : {day_share:.0f}%   (baseline)")
    print(f"whipsaw share of drawdown deepening : {xx:.0f}%   <- the [XX]%")
    print(f"concentration vs baseline           : {xx / day_share:.2f}x")
    print("-" * 60)
    print(f"max drawdown            : {out['max_dd']:,.1f}  "
          f"({out['max_dd_peak'].date()} -> {out['max_dd_trough'].date()})")
    if len(out["episode_shares"]) > 0:
        ep = 100 * out["episode_shares"].get("whipsaw", 0.0)
        print(f"whipsaw share of max-DD episode     : {ep:.0f}%")
    if out["n_unlabeled"]:
        print(f"(excluded {out['n_unlabeled']} ER warm-up days with no regime label)")
    print("=" * 60)

    daily = out["daily"].copy()
    daily["chop_score"] = score.reindex(daily.index)
    path = os.path.join(results_dir, f"regime_attribution_{MODE}.csv")
    daily.to_csv(path)
    print(f"saved per-day detail to results/regime_attribution_{MODE}.csv")
