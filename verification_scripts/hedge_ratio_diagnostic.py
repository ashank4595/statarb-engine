# hedge_ratio_diagnostic.py
# Read-only diagnostic. Touches nothing else in the repo.
#
# Question it answers: is the current TLS/PCA hedge ratio biased by the fact that
# PCA runs on RAW price levels? PCA finds the longest axis of the (A, B) point
# cloud. A stock trading at 28,000 has ~14x the rupee wobble of one trading at
# 2,000 even if their economic noise is identical -- so var(A) is ~196x var(B) in
# the covariance matrix purely because of scale. PCA reads that as signal, tilts
# the fitted line toward vertical, and beta comes out too large.
#
# Prints, for each pair, three betas plus the ADF p-value of the spread each one
# produces (a different beta => a different spread => a different p-value, which
# is what could change the tradeable universe):
#
#   tls_raw     - exactly what cointegration.hedge_ratio does today
#   ols         - the old asymmetric method, kept as a sanity anchor
#   tls_scaled  - proposed fix: standardize both legs, run PCA, rescale back
#
# How to read the output:
#   - tls_scaled ~= ols ~= tls_raw          -> scale bias is not biting. Do nothing.
#   - tls_raw >> ols, and tls_scaled ~= ols -> scale bias is real. Adopt the fix.
#   - principal[1] near 0                   -> the long axis is nearly vertical,
#                                              i.e. the failure mode, mid-flight.
#
# Run from the repo root: python3 -m candidate_pairs.hedge_ratio_diagnostic

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from data_layer import load_panel
from candidate_pairs.cointegration import spread_with_beta, adf_pvalue

FOLDER = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"

# (stock_a, stock_b). Ordered as the repo trades them: spread = A - beta * B.
# SHREECEM-ACC is the stress case: large price-level gap between the legs.
PAIRS = [
    ("SHREECEM", "ACC"),
    ("BAJAJFINSV", "CHOLAFIN"),
    ("SUNPHARMA", "LUPIN"),
    ("COALINDIA", "ONGC"),
    ("HAVELLS", "CROMPTON"),
]


def _align(price_a: pd.Series, price_b: pd.Series):
    """Align on date, drop incomplete rows, hand back plain float arrays."""
    combined = pd.concat([price_a, price_b], axis=1).dropna()
    a = combined.iloc[:, 0].to_numpy(dtype=float)
    b = combined.iloc[:, 1].to_numpy(dtype=float)
    return a, b


def tls_raw(a: np.ndarray, b: np.ndarray) -> float:
    """Current production hedge_ratio: PCA on raw price levels."""
    cov = np.cov(a, b)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    if np.isclose(principal[1], 0):
        return np.nan
    return float(principal[0] / principal[1])


def tls_scaled(a: np.ndarray, b: np.ndarray) -> float:
    """Proposed fix: standardize each leg so neither dominates the covariance
    matrix by virtue of its price level, run the same PCA, then rescale the
    slope back into rupee terms with std(A)/std(B).

    Keeps TLS's symmetry (A/B ordering still does not matter) and removes the
    scale sensitivity (which was never wanted)."""
    sa, sb = a.std(), b.std()
    if np.isclose(sa, 0) or np.isclose(sb, 0):
        return np.nan
    cov = np.cov(a / sa, b / sb)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    if np.isclose(principal[1], 0):
        return np.nan
    return float((principal[0] / principal[1]) * (sa / sb))


def ols(a: np.ndarray, b: np.ndarray) -> float:
    """The pre-TLS method. Asymmetric (A-on-B != B-on-A), which is exactly why it
    was replaced -- but it is immune to the scale problem, so it is a useful
    independent anchor for what beta ought to be roughly near."""
    model = sm.OLS(a, sm.add_constant(b)).fit()
    return float(model.params[1])


def diagnose(close: pd.DataFrame, stock_a: str, stock_b: str):
    """Compute all three betas for one pair, plus the covariance matrix and
    principal vector that produced the raw-TLS answer, plus the ADF p-value of
    the spread each beta implies."""
    if stock_a not in close.columns or stock_b not in close.columns:
        print(f"  skipping {stock_a}-{stock_b}: missing data")
        return None

    a, b = _align(close[stock_a], close[stock_b])
    if len(a) < 100:
        print(f"  skipping {stock_a}-{stock_b}: only {len(a)} overlapping days")
        return None

    cov = np.cov(a, b)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    betas = {
        "tls_raw": tls_raw(a, b),
        "ols": ols(a, b),
        "tls_scaled": tls_scaled(a, b),
    }

    # each beta implies a different spread -> a different ADF p-value.
    # this is the number that decides whether the pair is tradeable at all.
    pvals = {}
    for name, beta in betas.items():
        if np.isnan(beta):
            pvals[name] = np.nan
            continue
        s = spread_with_beta(close[stock_a], close[stock_b], beta).dropna()
        pvals[name] = adf_pvalue(s)

    print(f"\n{'=' * 68}")
    print(f"{stock_a} - {stock_b}   ({len(a)} overlapping days)")
    print(f"{'=' * 68}")
    print(f"  mean price      A = {a.mean():>10,.1f}     B = {b.mean():>10,.1f}"
          f"     ratio = {a.mean() / b.mean():.1f}x")
    print(f"  std dev         A = {a.std():>10,.1f}     B = {b.std():>10,.1f}"
          f"     ratio = {a.std() / b.std():.1f}x")
    print()
    print(f"  covariance matrix (raw prices):")
    print(f"      var(A)   = {cov[0, 0]:>14,.1f}")
    print(f"      cov(A,B) = {cov[0, 1]:>14,.1f}")
    print(f"      var(B)   = {cov[1, 1]:>14,.1f}")
    print(f"      var(A) / var(B) = {cov[0, 0] / cov[1, 1]:,.1f}x"
          f"   <- if this is huge, PCA is dominated by A's scale")
    print()
    print(f"  principal eigenvector = [dA={principal[0]:.6f}, dB={principal[1]:.6f}]")
    print(f"      beta = dA / dB = {principal[0]:.6f} / {principal[1]:.6f}"
          f" = {betas['tls_raw']:.3f}")
    if abs(principal[1]) < 0.05:
        print(f"      WARNING: dB is near zero -- long axis is nearly vertical,"
              f" beta is unstable here")
    print()
    print(f"  {'method':<14} {'beta':>12}   {'adf p-value':>12}   {'tradeable?':>10}")
    print(f"  {'-' * 14} {'-' * 12}   {'-' * 12}   {'-' * 10}")
    for name in ("tls_raw", "ols", "tls_scaled"):
        beta, p = betas[name], pvals[name]
        flag = "yes" if (not np.isnan(p) and p < 0.05) else "NO"
        print(f"  {name:<14} {beta:>12.3f}   {p:>12.4f}   {flag:>10}")

    ratio = betas["tls_raw"] / betas["ols"] if betas["ols"] else np.nan
    print()
    print(f"  tls_raw / ols = {ratio:.2f}x"
          f"   <- near 1.0 means no meaningful scale bias on this pair")

    return {
        "stock_a": stock_a, "stock_b": stock_b,
        "price_ratio": round(a.mean() / b.mean(), 1),
        "var_ratio": round(cov[0, 0] / cov[1, 1], 1),
        "beta_tls_raw": round(betas["tls_raw"], 3),
        "beta_ols": round(betas["ols"], 3),
        "beta_tls_scaled": round(betas["tls_scaled"], 3),
        "p_tls_raw": round(pvals["tls_raw"], 4),
        "p_ols": round(pvals["ols"], 4),
        "p_tls_scaled": round(pvals["tls_scaled"], 4),
        "raw_over_ols": round(ratio, 2),
    }


def symmetry_check(close: pd.DataFrame, stock_a: str, stock_b: str) -> None:
    """The reason TLS was adopted: hedge_ratio(A,B) should be 1/hedge_ratio(B,A).
    OLS fails this. Confirms the fix does NOT break the property you wanted."""
    a, b = _align(close[stock_a], close[stock_b])

    print(f"\n{'=' * 68}")
    print(f"symmetry check on {stock_a} - {stock_b}")
    print(f"  a symmetric estimator must satisfy  beta(A,B) == 1 / beta(B,A)")
    print(f"{'=' * 68}")
    print(f"  {'method':<14} {'beta(A,B)':>12} {'1/beta(B,A)':>14} {'symmetric?':>12}")
    print(f"  {'-' * 14} {'-' * 12} {'-' * 14} {'-' * 12}")
    for name, fn in (("tls_raw", tls_raw), ("ols", ols), ("tls_scaled", tls_scaled)):
        fwd = fn(a, b)
        rev = fn(b, a)
        inv = 1.0 / rev if rev else np.nan
        ok = "yes" if np.isclose(fwd, inv, rtol=1e-6) else "NO"
        print(f"  {name:<14} {fwd:>12.4f} {inv:>14.4f} {ok:>12}")


if __name__ == "__main__":
    tickers = sorted({t for pair in PAIRS for t in pair})
    print(f"loading {len(tickers)} tickers: {', '.join(tickers)}")
    close = load_panel(FOLDER, tickers=tickers)

    rows = []
    for stock_a, stock_b in PAIRS:
        row = diagnose(close, stock_a, stock_b)
        if row is not None:
            rows.append(row)

    symmetry_check(close, PAIRS[0][0], PAIRS[0][1])

    if rows:
        summary = pd.DataFrame(rows)
        print(f"\n{'=' * 68}")
        print("summary")
        print(f"{'=' * 68}")
        print(summary.to_string(index=False))
        print("\nread: if raw_over_ols is near 1.0 everywhere, the scale bias is")
        print("theoretical and the current hedge_ratio is fine. If it blows up on")
        print("the high price_ratio pairs, the bias is real and worth fixing.")
