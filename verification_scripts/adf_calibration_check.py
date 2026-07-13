# adf_calibration_check.py
# Is the ADF p-value even calibrated for what we are using it on?
#
# THE SUSPICION
# cointegration.py builds the spread as  A - beta*B  where beta was FITTED on the
# very same data, by minimizing that spread's variance (TLS/PCA). Then it hands the
# residual to adfuller() and reads off a p-value.
#
# But adfuller()'s critical values assume the series was HANDED to it -- not that
# somebody first searched for the linear combination that makes it look as
# stationary as possible. Fitting beta is a free optimization in the direction of
# stationarity, so the residual looks more mean-reverting than it earned. The
# p-values come out TOO SMALL. This is a known problem and it is exactly why
# Engle-Granger has its own critical values (Phillips-Ouliaris), which are stricter.
#
# statsmodels.tsa.stattools.coint() does the Engle-Granger test properly, with the
# right critical values. This script runs both on every pair and compares.
#
# THE EVIDENCE THAT PROMPTED THIS
# Under the null (no pair truly cointegrated), p-values are UNIFORM on [0, 1].
# The adfuller p-values on the first year were not remotely uniform -- only 23 pairs
# landed above p=0.50 where a uniform distribution predicts ~80. The whole
# distribution was squashed leftward, which is the signature of a miscalibrated
# test, not of real signal (real signal gives a SPIKE near zero while leaving the
# uniform tail intact).
#
# WHAT TO LOOK FOR
#   - if coint() p-values are systematically LARGER than adfuller()'s, the current
#     screen is optimistic and the tradeable universe is smaller than 29.
#   - if the coint() histogram is closer to uniform in the high bins, that confirms
#     adfuller was the thing distorting it.
#   - the pairs that survive BOTH tests are the ones actually worth trading.
#
# Read-only. Changes nothing.
# Run from the repo root: python3 -m verification_scripts.adf_calibration_check

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from data_layer import load_panel
from candidate_pairs.create_pairs import all_pairs
from candidate_pairs.cointegration import (
    hedge_ratio, correlation, pair_is_valid, spread_with_beta, adf_pvalue,
)

FOLDER = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"

FORMATION_MONTHS = 12
ALPHA = 0.05
FDR_Q = 0.10


def benjamini_hochberg(pvals: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask of which p-values survive BH-FDR control at level q.
    Sort ascending, keep everything up to the LARGEST i where p_(i) <= (i/n)*q."""
    n = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    thresholds = (np.arange(1, n + 1) / n) * q
    passing = np.where(sorted_p <= thresholds)[0]
    mask = np.zeros(n, dtype=bool)
    if len(passing):
        mask[order[: passing.max() + 1]] = True
    return mask


def histogram(pvals: pd.Series, label: str) -> None:
    """Print the p-value distribution against what UNIFORM would predict.

    This is the whole diagnostic. Under the null, p-values are uniform, so each bin
    should hold (bin_width * n) pairs. Real signal shows up as a SPIKE near zero
    with the rest of the distribution still roughly uniform. A distribution that is
    squashed leftward EVERYWHERE -- too few high p-values -- means the test itself
    is miscalibrated, not that you found something."""
    bins = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.01]
    labels = ["0.00-0.01", "0.01-0.05", "0.05-0.10", "0.10-0.25",
              "0.25-0.50", "0.50-1.00"]
    widths = [0.01, 0.04, 0.05, 0.15, 0.25, 0.50]
    n = len(pvals)

    counts = pd.cut(pvals, bins=bins, labels=labels,
                    right=False).value_counts().reindex(labels)

    print(f"\n  {label}   (n = {n})")
    print(f"  {'bin':>12}  {'observed':>8}  {'if uniform':>10}   distribution")
    print(f"  {'-'*12}  {'-'*8}  {'-'*10}   {'-'*30}")
    for lab, width in zip(labels, widths):
        obs = int(counts[lab])
        exp = width * n
        bar = "#" * min(obs, 40)
        flag = ""
        if lab == "0.50-1.00" and obs < 0.5 * exp:
            flag = "  <- far too few. test is miscalibrated."
        print(f"  {lab:>12}  {obs:>8}  {exp:>10.0f}   {bar}{flag}")


if __name__ == "__main__":
    pairs = all_pairs()
    tickers = sorted({t for _s, a, b in pairs for t in (a, b)})
    print(f"loading {len(tickers)} tickers...")
    close = load_panel(FOLDER, tickers=tickers)

    start = close.index.min()
    cutoff = start + pd.DateOffset(months=FORMATION_MONTHS)
    formation = close[close.index < cutoff]
    print(f"formation window: {start.date()} to {cutoff.date()} "
          f"({len(formation)} trading days)\n")

    rows = []
    for _sector, a, b in pairs:
        if a not in formation.columns or b not in formation.columns:
            continue
        pair = formation[[a, b]].dropna()
        if len(pair) < 100:
            continue

        beta = hedge_ratio(pair[a], pair[b])
        ok, _reason = pair_is_valid(pair[a], pair[b], beta)
        if not ok:
            continue

        # CURRENT METHOD: ADF on the residual of a beta we fitted ourselves.
        s = spread_with_beta(pair[a], pair[b], beta)
        p_adf = adf_pvalue(s)

        # CORRECT METHOD: Engle-Granger with Phillips-Ouliaris critical values,
        # which account for the fact that beta was estimated from this same data.
        # coint() runs its own OLS internally, so we do not pass our spread in.
        _tstat, p_coint, _crit = coint(pair[a].to_numpy(dtype=float),
                                       pair[b].to_numpy(dtype=float))

        rows.append({
            "pair": f"{a}-{b}",
            "corr": round(correlation(pair[a], pair[b]), 2),
            "beta": round(beta, 3),
            "p_adf": round(p_adf, 4),
            "p_coint": round(p_coint, 4),
            "ratio": round(p_coint / p_adf, 1) if p_adf > 0 else np.nan,
        })

    df = pd.DataFrame(rows).sort_values("p_adf").reset_index(drop=True)
    n = len(df)

    df["adf_0.05"] = df["p_adf"] < ALPHA
    df["coint_0.05"] = df["p_coint"] < ALPHA
    df["coint_bh"] = benjamini_hochberg(df["p_coint"].to_numpy(), FDR_Q)
    df["both"] = df["adf_0.05"] & df["coint_0.05"]

    pd.set_option("display.max_rows", None)
    print("=" * 84)
    print("ADF on a self-fitted residual  vs  proper Engle-Granger (coint)")
    print("=" * 84)
    print(df[["pair", "corr", "beta", "p_adf", "p_coint", "ratio",
              "adf_0.05", "coint_0.05", "coint_bh"]].to_string(index=False))

    n_adf = int(df["adf_0.05"].sum())
    n_coint = int(df["coint_0.05"].sum())
    n_both = int(df["both"].sum())
    n_bh = int(df["coint_bh"].sum())
    lost = int((df["adf_0.05"] & ~df["coint_0.05"]).sum())
    gained = int((~df["adf_0.05"] & df["coint_0.05"]).sum())

    print("\n" + "=" * 84)
    print("verdict")
    print("=" * 84)
    print(f"  pairs tested                          : {n}")
    print(f"  pass current ADF   (p < {ALPHA})         : {n_adf}")
    print(f"  pass proper coint  (p < {ALPHA})         : {n_coint}")
    print(f"  pass BOTH                             : {n_both}")
    print(f"  pass coint + BH-FDR (q = {FDR_Q})         : {n_bh}")
    print()
    print(f"  pairs the current screen WRONGLY accepts : {lost}")
    print(f"     ^ these clear ADF but fail proper Engle-Granger. They are in your")
    print(f"       book today and probably should not be.")
    print(f"  pairs the current screen wrongly rejects : {gained}")

    med_ratio = df["ratio"].median()
    print(f"\n  median  p_coint / p_adf  = {med_ratio:.1f}x")
    if med_ratio > 1.5:
        print(f"     ^ proper Engle-Granger p-values are systematically ~{med_ratio:.0f}x")
        print(f"       LARGER. Fitting beta on the same data you then test makes the")
        print(f"       spread look more stationary than it earned, so adfuller()")
        print(f"       reports p-values that are too small. The current screen is")
        print(f"       optimistic and lets junk into the book.")
    elif med_ratio < 0.7:
        print(f"     ^ coint() is LOOSER, which is unexpected. Worth investigating.")
    else:
        print(f"     ^ the two tests broadly agree. The calibration worry does not")
        print(f"       bite on this data, and the current screen is fine.")

    print("\n" + "=" * 84)
    print("p-value distributions: is the test calibrated?")
    print("=" * 84)
    print("\n  Under the null (nothing is cointegrated), p-values are UNIFORM.")
    print("  REAL SIGNAL   = spike near zero, tail stays roughly uniform.")
    print("  MISCALIBRATED = whole distribution squashed left, high bins starved.")

    histogram(df["p_adf"], "current: adfuller() on self-fitted residual")
    histogram(df["p_coint"], "proper: coint() with Engle-Granger critical values")

    if n_bh:
        print("\n" + "=" * 84)
        print(f"the {n_bh} pairs that survive proper Engle-Granger + BH-FDR")
        print("=" * 84)
        print(df[df["coint_bh"]][["pair", "corr", "beta", "p_adf", "p_coint"]]
              .to_string(index=False))
        print("\n  ^ this is the honest tradeable universe for year one.")
