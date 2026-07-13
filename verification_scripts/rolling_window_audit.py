# rolling_window_audit.py
# Answers two questions that the summary CSVs hide:
#
#   1. Is run_rolling actually re-selecting the pair universe every window?
#      i.e. test cointegration on trailing 24mo -> maybe 5 pairs pass -> trade those
#      5 for 3mo -> re-test -> maybe 2 pairs pass -> trade those 2. This prints the
#      surviving universe window by window so you can see it happen (or not).
#
#   2. Does the portfolio weighting make sense?
#      run_all_pairs pads each pair's daily P&L with 0.0 on every day that pair was
#      gated out. portfolio.py then weights by 1/std of that padded series. A pair
#      that traded 2 windows out of 6 is mostly zeros -> tiny std -> ENORMOUS weight.
#      Inverse-vol weighting cannot tell "flat because gated out" apart from
#      "genuinely low risk", so the pairs that barely trade get levered up the most.
#      This prints days-traded against weight-received so the relationship is visible.
#
# Read-only. Changes nothing.
# Run from the repo root: python3 -m verification_scripts.rolling_window_audit

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data_layer import load_panel
from candidate_pairs.create_pairs import all_pairs
from candidate_pairs.cointegration import hedge_ratio, correlation, pair_is_valid, \
    spread_with_beta, cointegration_pvalue, COINT_THRESHOLD, USE_ENGLE_GRANGER
from backtest.config import MODE, FORMATION_MONTHS, STEP_MONTHS, WEIGHT_SCHEME

FOLDER = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"

MIN_FORMATION_DAYS = 100


def audit_windows(close: pd.DataFrame, pairs: list,
                  formation_months: int, step_months: int) -> pd.DataFrame:
    """Replay run_rolling's window loop across ALL pairs at once, recording which
    pairs survive the gate in each window. This is the per-window universe."""
    start, end = close.index.min(), close.index.max()
    split = start + pd.DateOffset(months=formation_months)

    rows = []
    window_no = 0
    while split < end:
        window_no += 1
        trade_end = split + pd.DateOffset(months=step_months)
        form_start = split - pd.DateOffset(months=formation_months)

        survivors, reasons = [], {"low_corr": 0, "bad_beta": 0, "not_coint": 0,
                                  "short_data": 0}

        for _sector, a, b in pairs:
            if a not in close.columns or b not in close.columns:
                continue
            pair = close[[a, b]].dropna()
            formation = pair[(pair.index >= form_start) & (pair.index < split)]
            trading = pair[(pair.index >= split) & (pair.index < trade_end)]

            if len(formation) < MIN_FORMATION_DAYS or len(trading) < 5:
                reasons["short_data"] += 1
                continue

            beta = hedge_ratio(formation[a], formation[b])
            ok, reason = pair_is_valid(formation[a], formation[b], beta)
            if not ok:
                if "corr" in reason:
                    reasons["low_corr"] += 1
                else:
                    reasons["bad_beta"] += 1
                continue

            # use the SAME test the pipeline now uses -- proper Engle-Granger with
            # Phillips-Ouliaris critical values, not adfuller() on the fitted residual.
            if cointegration_pvalue(formation[a], formation[b], beta) >= COINT_THRESHOLD:
                reasons["not_coint"] += 1
                continue

            survivors.append(f"{a}-{b}")

        rows.append({
            "window": window_no,
            "formation": f"{form_start.date()} to {split.date()}",
            "trading": f"{split.date()} to {min(trade_end, end).date()}",
            "passed": len(survivors),
            "low_corr": reasons["low_corr"],
            "bad_beta": reasons["bad_beta"],
            "not_coint": reasons["not_coint"],
            "survivors": survivors,
        })
        split = trade_end

    return pd.DataFrame(rows)


def audit_weights() -> None:
    """Compare how many days each pair ACTUALLY traded against the weight
    portfolio.py handed it. If the relationship is inverse, the zero-padding is
    driving the book."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "results", f"daily_pnl_{MODE}.csv")
    if not os.path.exists(path):
        print(f"\n  {path} not found -- run backtest.run_all_pairs first")
        return

    daily = pd.read_csv(path, index_col=0, parse_dates=True)

    # a day the pair was gated out was padded with exactly 0.0 by run_all_pairs.
    # a day it held a position but the spread did not move is vanishingly rare,
    # so non-zero days is a good proxy for days actually traded.
    active_days = (daily != 0).sum()
    total_days = len(daily)

    # CONFIRMED BUG, now fixed in portfolio.py: vol must be measured only on days the
    # pair actually held a position. The padded zeros shrink measured vol by roughly
    # sqrt(fraction active), so a pair live 8% of the time looks ~3.5x safer than it is
    # and 1/vol levers it up accordingly. Both are shown so the size of the distortion
    # is visible.
    vol_padded = daily.std().replace(0, pd.NA).dropna()
    w_padded = 1.0 / vol_padded
    w_padded = w_padded / w_padded.sum() * len(w_padded)

    vol_active = daily[daily != 0].std().replace(0, pd.NA).dropna()
    w_active = 1.0 / vol_active
    w_active = w_active / w_active.sum() * len(w_active)

    table = pd.DataFrame({
        "active_days": active_days,
        "pct_active": (100 * active_days / total_days).round(1),
        "vol_padded": vol_padded.round(4),
        "vol_active": vol_active.round(4),
        "weight_OLD": w_padded.round(2),
        "weight_NEW": w_active.round(2),
    }).dropna().sort_values("weight_OLD", ascending=False)

    print("\n" + "=" * 88)
    print(f"portfolio weights vs days actually traded   (scheme = {WEIGHT_SCHEME})")
    print("=" * 88)
    print(table.to_string())

    corr_old = table["weight_OLD"].corr(table["active_days"])
    corr_new = table["weight_NEW"].corr(table["active_days"])
    print(f"\n  correlation(weight_OLD, active_days) = {corr_old:+.3f}")
    print(f"  correlation(weight_NEW, active_days) = {corr_new:+.3f}")
    if corr_old < -0.2:
        print("\n  weight_OLD is NEGATIVELY correlated with activity: the less a pair")
        print("  traded, the MORE weight it got. Inverse-vol was reading zero-padded")
        print("  (gated-out) days as 'low risk' and levering those pairs up.")
        print("  weight_NEW masks the zeros and measures risk-when-deployed instead.")

    top = table.head(6)
    print(f"\n  the {len(top)} highest-weighted pairs traded on average "
          f"{top['active_days'].mean():.0f} of {total_days} days "
          f"({100 * top['active_days'].mean() / total_days:.0f}%)")
    bottom = table.tail(6)
    print(f"  the {len(bottom)} lowest-weighted pairs traded on average "
          f"{bottom['active_days'].mean():.0f} of {total_days} days "
          f"({100 * bottom['active_days'].mean() / total_days:.0f}%)")


if __name__ == "__main__":
    pairs = all_pairs()
    tickers = sorted({t for _s, a, b in pairs for t in (a, b)})
    print(f"loading {len(tickers)} tickers...")
    close = load_panel(FOLDER, tickers=tickers)

    print(f"\nreplaying rolling windows: FORMATION_MONTHS={FORMATION_MONTHS}, "
          f"STEP_MONTHS={STEP_MONTHS}, {len(pairs)} candidate pairs")
    print(f"cointegration test: "
          f"{'Engle-Granger (Phillips-Ouliaris)' if USE_ENGLE_GRANGER else 'ADF on fitted residual'}")

    df = audit_windows(close, pairs, FORMATION_MONTHS, STEP_MONTHS)

    print("\n" + "=" * 74)
    print("per-window universe: how many pairs passed the gate each quarter")
    print("=" * 74)
    print(df.drop(columns=["survivors"]).to_string(index=False))

    print("\n" + "=" * 74)
    print("which pairs survived each window")
    print("=" * 74)
    for _, row in df.iterrows():
        names = ", ".join(row["survivors"]) if row["survivors"] else "(none)"
        print(f"\nwindow {row['window']}  trade {row['trading']}  "
              f"-> {row['passed']} pairs")
        print(f"  {names}")

    # does the universe actually CHANGE window to window, or is it the same pairs?
    sets = [set(s) for s in df["survivors"]]
    if len(sets) > 1:
        print("\n" + "=" * 74)
        print("universe churn between consecutive windows")
        print("=" * 74)
        for i in range(1, len(sets)):
            added = sets[i] - sets[i - 1]
            dropped = sets[i - 1] - sets[i]
            kept = sets[i] & sets[i - 1]
            print(f"  window {i} -> {i+1}:  kept {len(kept)}, "
                  f"added {len(added)}, dropped {len(dropped)}")
        stable = set.intersection(*sets) if all(sets) else set()
        print(f"\n  pairs that passed in EVERY window: {len(stable)}")
        if stable:
            print(f"    {', '.join(sorted(stable))}")

    audit_weights()
