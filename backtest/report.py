# report.py
# One-page portfolio report: summary stats + equity curve + a colored
# monthly returns (%) matrix for the whole portfolio.
#
# Reads: results/daily_pnl_{MODE}.csv, results/portfolio_{MODE}_{WEIGHT_SCHEME}.csv
# Also saves: results/monthly_returns_pct_{MODE}_{WEIGHT_SCHEME}.csv,
#             results/per_pair_quarterly_sharpe_{MODE}.csv (saved, not plotted --
#             still there for spotting a pair that blew up in one quarter)
# Run: python3 -m backtest.report

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from backtest.evaluate import max_drawdown, annualized_return

MODE = "split"
WEIGHT_SCHEME = "equal_risk"


def monthly_matrix(pnl: pd.Series) -> pd.DataFrame:
    """Monthly net P&L in rupees, rows=year, columns=Jan..Dec + Total + Average."""
    df = pnl.to_frame("pnl")
    df["year"], df["month"] = df.index.year, df.index.month
    grid = df.pivot_table(index="year", columns="month", values="pnl", aggfunc="sum")
    grid.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in grid.columns]
    grid["Total"] = grid.sum(axis=1)
    grid.loc["Average"] = grid.mean(axis=0)
    return grid


def per_pair_quarterly_sharpe(daily_pnl: pd.DataFrame) -> pd.DataFrame:
    """Pairs x quarters grid, saved to CSV but not plotted here -- catches a
    pair that's profitable overall but blew up in a single quarter."""
    q_pnl = daily_pnl.resample("QE").sum()
    q_days = daily_pnl.resample("QE").apply(lambda col: (col != 0).sum())
    daily_std = daily_pnl.std().replace(0, np.nan)
    q_vol = (q_days ** 0.5).multiply(daily_std, axis=1)
    score = q_pnl / q_vol.replace(0, np.nan)
    score.index = score.index.to_period("Q").astype(str)
    return score.T


def summary_stats(equity: pd.Series, daily_pnl: pd.Series, total_capital: float) -> dict:
    total_pnl = equity.iloc[-1]
    years = len(equity) / 252
    max_dd = max_drawdown(equity)
    ann_ret_pct = annualized_return(equity, total_capital) * 100
    ann_ret_rupees = total_pnl / years
    calmar = (total_pnl / years) / abs(max_dd) if max_dd != 0 else float("nan")
    win_days = int((daily_pnl > 0).sum())
    total_days = int((daily_pnl != 0).sum())
    return {
        "Annual Return": f"{ann_ret_rupees:,.0f}  ({ann_ret_pct:.2f}%)",
        "Total P&L": f"{total_pnl:,.0f}",
        "Max Drawdown": f"{max_dd:,.0f}",
        "Calmar Ratio": f"{calmar:.2f}",
        "Win / Total Days": f"{win_days} / {total_days} ({100*win_days/total_days:.1f}%)" if total_days else "n/a",
    }


def build_report():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(root, "results")

    daily_pnl = pd.read_csv(os.path.join(results_dir, f"daily_pnl_{MODE}.csv"),
                             index_col=0, parse_dates=True)
    portfolio = pd.read_csv(os.path.join(results_dir, f"portfolio_{MODE}_{WEIGHT_SCHEME}.csv"),
                             index_col=0, parse_dates=True)
    total_capital = float(portfolio["total_capital"].iloc[0])

    monthly = monthly_matrix(portfolio["net_pnl"])
    monthly_pct = monthly / total_capital * 100

    per_pair_q = per_pair_quarterly_sharpe(daily_pnl)
    per_pair_q.to_csv(os.path.join(results_dir, f"per_pair_quarterly_sharpe_{MODE}.csv"))

    stats = summary_stats(portfolio["equity"], portfolio["net_pnl"], total_capital)

    monthly_pct.to_csv(os.path.join(results_dir, f"monthly_returns_pct_{MODE}_{WEIGHT_SCHEME}.csv"))
    print("saved monthly_returns_pct / per_pair_quarterly_sharpe CSVs to results/")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.3], hspace=0.5, wspace=0.3)

    ax_stats = fig.add_subplot(gs[0, 0]); ax_stats.axis("off")
    ax_stats.set_title("Portfolio Summary", fontsize=12, loc="left")
    ax_stats.table(cellText=[[v] for v in stats.values()], rowLabels=list(stats.keys()),
                   colLabels=["Value"], loc="center", cellLoc="left").scale(1, 1.6)

    ax_eq = fig.add_subplot(gs[0, 1])
    portfolio["equity"].plot(ax=ax_eq, color="steelblue")
    ax_eq.axhline(0, color="black", alpha=0.3)
    ax_eq.set_title(f"Portfolio Equity Curve ({MODE}, {WEIGHT_SCHEME})")

    ax_month = fig.add_subplot(gs[1, :])
    ax_month.set_title("Monthly Returns Matrix (% of capital)", fontsize=12, loc="left")
    vals = monthly_pct.values.astype(float)
    finite = vals[~np.isnan(vals)]
    vmax = max(1.0, np.abs(finite).max()) if finite.size else 1.0
    im = ax_month.imshow(vals, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax_month.set_xticks(range(len(monthly_pct.columns)))
    ax_month.set_xticklabels(monthly_pct.columns, fontsize=9)
    ax_month.set_yticks(range(len(monthly_pct.index)))
    ax_month.set_yticklabels(monthly_pct.index, fontsize=9)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isnan(v):
                ax_month.text(j, i, f"{v:.2f}%", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax_month, label="% return", fraction=0.02, pad=0.01)

    out_png = os.path.join(results_dir, f"report_{MODE}_{WEIGHT_SCHEME}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved report to results/report_{MODE}_{WEIGHT_SCHEME}.png")
    plt.show()


if __name__ == "__main__":
    build_report()
