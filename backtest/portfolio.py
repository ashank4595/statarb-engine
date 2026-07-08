# portfolio.py
# Combines multiple pairs' daily P&L (results/daily_pnl_{MODE}.csv) into one
# portfolio equity curve, equal-risk weighted.
#
# CAVEAT: weights are computed from the full trading-period realized vol of
# each pair's own P&L (the same data being weighted) -- a mild look-ahead in
# the WEIGHTING step only. Pair selection, hedge ratios, entry/exit signals
# are untouched and remain formation-only.
#
# Also computes total_capital = sum(weight_i * margin_i) across the pairs
# actually combined, using each pair's margin from results_{MODE}.csv. That's
# the capital base report.py needs to turn portfolio P&L into a % return.
#
# Run: python3 -m backtest.portfolio

import os
import pandas as pd

MODE = "split"                # must match a saved daily_pnl_{MODE}.csv
WEIGHT_SCHEME = "equal_risk"  # "equal_weight" | "equal_risk"


def combine(daily_pnl: pd.DataFrame, margins: pd.Series, scheme: str = "equal_risk"):
    if scheme == "equal_weight":
        weights = pd.Series(1.0, index=daily_pnl.columns)
    elif scheme == "equal_risk":
        vol = daily_pnl.std().replace(0, pd.NA).dropna()
        weights = 1.0 / vol
        weights = weights / weights.sum() * len(weights)   # avg weight = 1
    else:
        raise ValueError(f"unknown WEIGHT_SCHEME: {scheme}")

    weighted = daily_pnl[weights.index].multiply(weights, axis=1)
    portfolio = pd.DataFrame({"net_pnl": weighted.sum(axis=1)})
    portfolio["equity"] = portfolio["net_pnl"].cumsum()

    # capital base: each pair's own margin, scaled by how much weight it got --
    # a pair upweighted 2x is treated as if you deployed 2x its margin.
    pair_margins = margins.reindex(weights.index)
    total_capital = (weights * pair_margins).sum()

    portfolio.attrs["weights"] = weights
    portfolio.attrs["total_capital"] = total_capital
    return portfolio


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(root, "results")

    daily_pnl = pd.read_csv(os.path.join(results_dir, f"daily_pnl_{MODE}.csv"),
                             index_col=0, parse_dates=True)

    summary = pd.read_csv(os.path.join(root, f"results_{MODE}.csv"))
    summary["pair"] = summary["stock_a"] + "-" + summary["stock_b"]
    margins = summary.set_index("pair")["margin"]

    portfolio = combine(daily_pnl, margins, scheme=WEIGHT_SCHEME)

    out_df = portfolio[["net_pnl", "equity"]].copy()
    out_df["total_capital"] = portfolio.attrs["total_capital"]   # constant column so report.py can read it back
    out = os.path.join(results_dir, f"portfolio_{MODE}_{WEIGHT_SCHEME}.csv")
    out_df.to_csv(out)
    print(f"saved portfolio equity to results/portfolio_{MODE}_{WEIGHT_SCHEME}.csv")
    print(f"total capital (equal-risk weighted): {portfolio.attrs['total_capital']:,.0f}")
    print("\nweights used:")
    print(portfolio.attrs["weights"].sort_values(ascending=False).round(2).to_string())
