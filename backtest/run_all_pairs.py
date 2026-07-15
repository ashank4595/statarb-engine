# run_all_pairs.py
# Tests every pair from create_pairs.all_pairs() and reports how many are
# profitable. MODE selects the validation method (logic lives in validation_methods.py):
#   "full"    - fit and trade on all data (in-sample, optimistic)
#   "split"   - fit on formation period, trade the later period once
#   "rolling" - re-fit every STEP_MONTHS and trade forward
# Also persists each tradeable pair's daily net_pnl to results/daily_pnl_{MODE}.csv
# so portfolio.py and report.py can combine / analyze them later.
# Run: python3 -m backtest.run_all_pairs

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data_layer import load_panel
from candidate_pairs.create_pairs import all_pairs
from backtest.validation_methods import run_full, run_split, run_rolling
from backtest.config import MODE, FREQ   # single source of truth -- see config.py

FOLDER = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"

TESTERS = {"full": run_full, "split": run_split, "rolling": run_rolling}


if __name__ == "__main__":
    tester = TESTERS[MODE]
    pairs = all_pairs()
    tickers = sorted({t for _, a, b in pairs for t in [a, b]})
    print(f"loading {len(tickers)} tickers, mode = {MODE}...")
    close = load_panel(FOLDER, tickers=tickers, freq=FREQ)

    print(f"testing {len(pairs)} candidate pairs...\n")
    rows = []
    daily_pnl = {}   # pair_name -> daily net_pnl Series (recovered from equity.diff())

    for sector, a, b in pairs:
        if a not in close.columns or b not in close.columns:
            continue
        out = tester(close, a, b)
        if out is None:
            continue
        sh, pnl, equity, margin = out
        rows.append({
            "stock_a": a, "stock_b": b,
            "sharpe": round(sh, 2), "net_pnl": round(pnl, 1),
            "profitable": sh > 0,
            "margin": round(margin, 1),
        })
        # equity is cumsum(net_pnl), so diff() recovers the daily series exactly
        daily = equity.diff()
        daily.iloc[0] = equity.iloc[0]
        daily_pnl[f"{a}-{b}"] = daily

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

    root = os.path.dirname(os.path.dirname(__file__))
    results.to_csv(os.path.join(root, f"results_{MODE}.csv"), index=False)
    print(f"\nsaved summary to results_{MODE}.csv")

    # wide DataFrame: index=date, one column per pair. Pairs can have different
    # tradeable date ranges (esp. rolling mode) -> outer-join on union of dates,
    # fill gaps with 0 (no position = no P&L that day, not "unknown").
    daily_pnl_df = pd.concat(daily_pnl, axis=1).fillna(0.0).sort_index()
    daily_dir = os.path.join(root, "results")
    os.makedirs(daily_dir, exist_ok=True)
    daily_pnl_df.to_csv(os.path.join(daily_dir, f"daily_pnl_{MODE}.csv"))
    print(f"saved daily P&L per pair to results/daily_pnl_{MODE}.csv")