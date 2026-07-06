# run_all_pairs.py
# Tests every pair from create_pairs.all_pairs() and reports how many are
# profitable. MODE selects the validation method:
#   "full"    - fit and trade on all data (in-sample, optimistic) 
#   "split"   - fit on formation period, trade the later period once, using walk_forward.py
#   "rolling" - re-fit every STEP_MONTHS and trade forward, using walk_forward_rolling.py
# Run: python3 -m backtest.run_all_pairs

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data_layer import load_panel
from candidate_pairs.create_pairs import all_pairs
from candidate_pairs.cointegration import hedge_ratio, spread, spread_with_beta, adf_pvalue
from backtest.zscore_signal import zscore, positions
from backtest.engine import backtest_pair
from backtest.evaluate import sharpe

FOLDER = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"

# --- choose the validation method ---
MODE = "split"                 # "full" | "split" | "rolling"
SPLIT_DATE = "2025-01-01"      # boundary for "split" mode
FORMATION_MONTHS = 24          # formation length for "rolling" mode
STEP_MONTHS = 3                # trade-forward step for "rolling" mode
COINT_THRESHOLD = 0.05


def test_full(close, a, b):
    """Fit and trade on all data (in-sample)."""
    s = spread(close[a], close[b])
    if len(s) < 100 or adf_pvalue(s) >= COINT_THRESHOLD:
        return None
    res = backtest_pair(s, positions(zscore(s)))
    margin = close[a].mean() * 2 * 0.20
    return sharpe(res["net_pnl"] / margin), res["net_pnl"].sum()


def test_split(close, a, b):
    """Fit on formation, trade the later period once."""
    formation = close[close.index < SPLIT_DATE]
    trading = close[close.index >= SPLIT_DATE]

    form_spread = spread(formation[a], formation[b])
    if len(form_spread) < 100 or adf_pvalue(form_spread) >= COINT_THRESHOLD:
        return None

    beta = hedge_ratio(formation[a].dropna(), formation[b].dropna())
    trade_spread = spread_with_beta(trading[a], trading[b], beta)
    if len(trade_spread) < 65:
        return None
    res = backtest_pair(trade_spread, positions(zscore(trade_spread)))
    margin = trading[a].mean() * 2 * 0.20
    return sharpe(res["net_pnl"] / margin), res["net_pnl"].sum()


def test_rolling(close, a, b):
    """Re-fit every STEP_MONTHS and trade forward, stitching the P&L."""
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
    margin = pair[a].mean() * 2 * 0.20
    return sharpe(stitched["net_pnl"] / margin), stitched["net_pnl"].sum()


TESTERS = {"full": test_full, "split": test_split, "rolling": test_rolling}


if __name__ == "__main__":
    tester = TESTERS[MODE]
    pairs = all_pairs()
    tickers = sorted({t for _, a, b in pairs for t in [a, b]})
    print(f"loading {len(tickers)} tickers, mode = {MODE}...")
    close = load_panel(FOLDER, tickers=tickers)

    print(f"testing {len(pairs)} candidate pairs...\n")
    rows = []
    for sector, a, b in pairs:
        if a not in close.columns or b not in close.columns:
            continue
        out = tester(close, a, b)
        if out is None:
            continue
        sh, pnl = out
        rows.append({
            "stock_a": a, "stock_b": b,
            "sharpe": round(sh, 2), "net_pnl": round(pnl, 1),
            "profitable": sh > 0,
        })

    results = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

    n = len(results)
    n_profit = int(results["profitable"].sum())
    print("=" * 55)
    print(f"mode                : {MODE}")
    print(f"pairs tradeable     : {n}")
    print(f"pairs profitable    : {n_profit}  ({100*n_profit/n:.0f}%)")
    print("=" * 55)
    print(results.to_string(index=False))

    profitable = results[results["profitable"]]
    if len(profitable) > 0:
        print(f"\nmean Sharpe of profitable pairs: {profitable['sharpe'].mean():.2f}")

    out = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"results_{MODE}.csv")
    results.to_csv(out, index=False)
    print(f"\nsaved to results_{MODE}.csv")
