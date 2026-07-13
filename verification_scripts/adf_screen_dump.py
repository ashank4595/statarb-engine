# adf_screen_dump.py
# Prints the ADF p-value for EVERY candidate pair, fitted on the FIRST YEAR of data
# only, so you can see with your own eyes how many clear the 0.05 threshold -- and
# how many of those are expected to be pure luck.
#
# THE MULTIPLE TESTING PROBLEM, PLAINLY
#   ADF asks: "is this spread just random noise?"
#   The p-value answers: "if it WERE pure noise, how often would luck alone make it
#   look this mean-reverting?"
#   p < 0.05 means "less than 1 time in 20".
#
#   But you test ~322 pairs. Rolling a 20-sided die 322 times, you expect ~16 ones.
#   So even if NOT ONE pair is truly cointegrated -- if every spread is pure noise --
#   roughly 16 will still pass p < 0.05 by luck. That is not a bug. That is what a 5%
#   false-positive rate means when applied 322 times.
#
#   If your screen returns ~16 pairs, you have found nothing at all.
#   If it returns 60, maybe ~44 are real -- but you cannot tell WHICH 16 are junk.
#
# TWO STANDARD CORRECTIONS, both reported below:
#   Bonferroni  - divide the threshold by the number of tests: 0.05 / 322 = 0.000155.
#                 Controls the chance of even ONE false positive. Very conservative.
#   BH-FDR      - Benjamini-Hochberg. Controls the FRACTION of accepted pairs that are
#                 false positives (the "false discovery rate"). Sort p-values ascending;
#                 keep pair i if p_i <= (i / n) * q. Standard choice in finance.
#
# Read-only. Changes nothing.
# Run from the repo root: python3 -m verification_scripts.adf_screen_dump

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data_layer import load_panel
from candidate_pairs.create_pairs import all_pairs
from candidate_pairs.cointegration import (
    hedge_ratio, correlation, pair_is_valid, spread_with_beta, adf_pvalue,
    MIN_CORRELATION,
)

FOLDER = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"

FORMATION_MONTHS = 12     # "the first year"
ALPHA = 0.05              # the conventional per-pair threshold
FDR_Q = 0.10              # Benjamini-Hochberg target false discovery rate


def benjamini_hochberg(pvals: np.ndarray, q: float) -> np.ndarray:
    """Return a boolean mask of which p-values survive BH-FDR control at level q.

    Sort ascending. Find the LARGEST i where p_(i) <= (i/n)*q. Keep everything up to
    and including it. Unlike Bonferroni (which asks 'what is the chance of ANY false
    positive?'), this asks 'what fraction of the pairs I accept are false?' -- a much
    more useful question when you expect some real signal to exist.
    """
    n = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    thresholds = (np.arange(1, n + 1) / n) * q
    passing = np.where(sorted_p <= thresholds)[0]

    mask = np.zeros(n, dtype=bool)
    if len(passing) == 0:
        return mask
    cutoff_rank = passing.max()          # largest index that satisfies the condition
    mask[order[: cutoff_rank + 1]] = True
    return mask


if __name__ == "__main__":
    pairs = all_pairs()
    tickers = sorted({t for _s, a, b in pairs for t in (a, b)})
    print(f"loading {len(tickers)} tickers...")
    close = load_panel(FOLDER, tickers=tickers)

    start = close.index.min()
    cutoff = start + pd.DateOffset(months=FORMATION_MONTHS)
    formation = close[close.index < cutoff]

    print(f"\nformation window: {start.date()} to {cutoff.date()} "
          f"({len(formation)} trading days, {FORMATION_MONTHS} months)")
    print(f"testing {len(pairs)} candidate pairs\n")

    rows = []
    gated_out = 0
    for _sector, a, b in pairs:
        if a not in formation.columns or b not in formation.columns:
            continue
        pair = formation[[a, b]].dropna()
        if len(pair) < 100:
            continue

        beta = hedge_ratio(pair[a], pair[b])
        corr = correlation(pair[a], pair[b])
        ok, reason = pair_is_valid(pair[a], pair[b], beta)

        if not ok:
            gated_out += 1
            rows.append({
                "pair": f"{a}-{b}", "corr": round(corr, 2) if not np.isnan(corr) else None,
                "beta": round(beta, 3) if not np.isnan(beta) else None,
                "adf_p": None, "gate": "REJECTED", "reason": reason,
            })
            continue

        s = spread_with_beta(pair[a], pair[b], beta)
        p = adf_pvalue(s)
        rows.append({
            "pair": f"{a}-{b}", "corr": round(corr, 2), "beta": round(beta, 3),
            "adf_p": round(p, 4), "gate": "ok", "reason": "",
        })

    df = pd.DataFrame(rows)
    tested = df[df["gate"] == "ok"].copy().sort_values("adf_p").reset_index(drop=True)

    print("=" * 78)
    print(f"ADF p-value for every pair that passed the gate  "
          f"(|corr| >= {MIN_CORRELATION}, beta > 0)")
    print("=" * 78)

    n = len(tested)
    pvals = tested["adf_p"].to_numpy()

    bonferroni_alpha = ALPHA / n
    tested["passes_0.05"] = pvals < ALPHA
    tested["passes_bonf"] = pvals < bonferroni_alpha
    tested["passes_bh"] = benjamini_hochberg(pvals, FDR_Q)

    pd.set_option("display.max_rows", None)
    print(tested[["pair", "corr", "beta", "adf_p",
                  "passes_0.05", "passes_bonf", "passes_bh"]].to_string(index=False))

    n_005 = int(tested["passes_0.05"].sum())
    n_bonf = int(tested["passes_bonf"].sum())
    n_bh = int(tested["passes_bh"].sum())
    expected_false = ALPHA * n

    print("\n" + "=" * 78)
    print("the multiple testing problem, in your own numbers")
    print("=" * 78)
    print(f"  candidate pairs                     : {len(rows)}")
    print(f"  rejected by the gate before ADF     : {gated_out}")
    print(f"  actually tested with ADF            : {n}")
    print()
    print(f"  pass p < {ALPHA}                        : {n_005}")
    print(f"  EXPECTED to pass by pure luck        : {expected_false:.0f}"
          f"   ({ALPHA:.0%} of {n} tests)")
    print()
    if n_005 <= expected_false:
        print(f"  >> {n_005} passed and ~{expected_false:.0f} were expected from noise alone.")
        print(f"     This screen has found NOTHING distinguishable from chance.")
    else:
        signal = n_005 - expected_false
        print(f"  >> {n_005} passed, ~{expected_false:.0f} expected from noise.")
        print(f"     At most ~{signal:.0f} are plausibly real -- but you cannot tell")
        print(f"     WHICH ~{expected_false:.0f} of your {n_005} are the junk.")
    print()
    print(f"  survive Bonferroni (p < {bonferroni_alpha:.6f}) : {n_bonf}")
    print(f"     ^ controls the chance of even ONE false positive. Brutal but honest.")
    print(f"  survive BH-FDR at q = {FDR_Q}          : {n_bh}")
    print(f"     ^ allows ~{FDR_Q:.0%} of the accepted pairs to be false. The usual")
    print(f"       choice, and the one worth quoting in an interview.")

    print("\n" + "=" * 78)
    print("p-value distribution")
    print("=" * 78)
    bins = [0, 0.0001, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.01]
    labels = ["<0.0001", "0.0001-0.001", "0.001-0.01", "0.01-0.05",
              "0.05-0.10", "0.10-0.25", "0.25-0.50", "0.50-1.00"]
    counts = pd.cut(tested["adf_p"], bins=bins, labels=labels,
                    right=False).value_counts().reindex(labels)
    for label, count in counts.items():
        bar = "#" * int(count) if count == count else ""
        print(f"  {label:>14}  {int(count):>4}  {bar}")

    print("\n  NOTE: under the null (no pair is truly cointegrated), p-values are")
    print("  UNIFORMLY distributed -- you would see roughly equal counts in every")
    print("  equal-width bin. A big pile-up near zero is evidence of real signal.")
    print("  A flat spread is evidence there is nothing here.")

    if gated_out:
        print("\n" + "=" * 78)
        print(f"the {gated_out} pairs rejected by the gate before ADF ever ran")
        print("=" * 78)
        print(df[df["gate"] == "REJECTED"][["pair", "corr", "beta", "reason"]]
              .to_string(index=False))
