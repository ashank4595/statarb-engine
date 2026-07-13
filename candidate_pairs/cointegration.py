# Must run this as a module:
# statarb-engine % source venv/bin/activate
# (venv) statarb-engine % python3 -m candidate_pairs.cointegration

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- tunable constants -------------------------------------------------------

# A pair whose legs do not move together has no relationship to trade. Below this
# correlation the hedge ratio is fitted to noise and comes out meaningless -- see
# SHREECEM-ACC, which had corr = -0.07 and produced beta = -103 under raw TLS.
# No estimator can rescue that; the pair must be rejected before beta is used.
MIN_CORRELATION = 0.5

# Reject non-positive beta on a same-sector pair. spread = A - beta*B, so a
# negative beta means spread = A + |beta|*B: long BOTH legs, i.e. leveraged
# directional exposure, not a market-neutral spread. If the fit says that, the
# fit has failed.
MIN_BETA = 0.0

COINT_THRESHOLD = 0.05   # ADF p-value a spread must beat to be called mean-reverting


# --- hedge ratio -------------------------------------------------------------

def correlation(price_a: pd.Series, price_b: pd.Series) -> float:
    """Pearson correlation of the two price levels, on their overlapping days.

    This is the screen that should run BEFORE the hedge ratio. Cointegration
    tests and PCA both assume there is a relationship to find; neither one
    errors out when there isn't, they just return numbers.

    @return correlation in [-1, 1], or nan if fewer than 2 overlapping days.
    """
    combined = pd.concat([price_a, price_b], axis=1).dropna()
    if len(combined) < 2:
        return np.nan
    return float(combined.iloc[:, 0].corr(combined.iloc[:, 1]))


def hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """
    Scale-corrected TLS/PCA hedge ratio for:
        spread = A - beta * B
    Returns beta such that A ~= beta * B.

    HOW IT WORKS
    PCA finds the longest axis of the (A, B) point cloud and reports its slope.
    That axis is the eigenvector of the covariance matrix with the largest
    eigenvalue.

    WHY THE STANDARDIZATION STEP
    PCA is not scale-invariant. A stock at 26,000 has ~12x the rupee wobble of
    one at 2,100 even when their economic noise is identical, so var(A) enters
    the covariance matrix ~150x larger purely because of the price level. PCA
    cannot tell "this direction has more real variation" apart from "this
    direction is measured in bigger numbers" -- it reads the inflated variance
    as signal, tilts the fitted line toward vertical, and returns a beta that is
    too large. Measured on real data: HAVELLS-CROMPTON came out 42% inflated
    (4.44 vs 3.12 OLS); SHREECEM-ACC came out at -103.

    Dividing each leg by its own std forces var(A) = var(B) = 1 in the matrix, so
    neither stock can dominate on price level alone. Multiplying the resulting
    slope by std(A)/std(B) puts it back into rupee terms.

    WHY NOT OLS
    OLS minimizes vertical distance only, treating B as noise-free. Both legs are
    noisy, and the consequence is that OLS is asymmetric: hedge_ratio(A,B) is not
    1/hedge_ratio(B,A), so the arbitrary choice of which stock is "A" changes the
    spread being traded. TLS minimizes perpendicular distance and is symmetric.
    Standardizing preserves that symmetry while removing the scale sensitivity.

    @return beta, or nan if the fit is degenerate (flat series, vertical axis).
    """
    combined = pd.concat([price_a, price_b], axis=1).dropna()

    if len(combined) < 2:
        return np.nan

    a = combined.iloc[:, 0].to_numpy(dtype=float)
    b = combined.iloc[:, 1].to_numpy(dtype=float)

    # standardize: strip the price-level scale off both legs so the covariance
    # matrix reflects co-movement, not which stock has bigger numbers.
    sa, sb = a.std(), b.std()
    if np.isclose(sa, 0) or np.isclose(sb, 0):
        return np.nan   # a flat series has no direction to find

    cov = np.cov(a / sa, b / sb)

    # eigh is specifically for real symmetric matrices, which covariance always is.
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigenvectors come back as COLUMNS. Column of the largest eigenvalue is the
    # long axis of the cloud -- the direction of greatest shared movement.
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    # principal = [dA, dB]: "move along the long axis and A changes by dA while
    # B changes by dB". spread is A - beta*B, so beta = dA/dB.
    if np.isclose(principal[1], 0):
        return np.nan

    # rescale from standardized units back to rupees.
    return float((principal[0] / principal[1]) * (sa / sb))


def pair_is_valid(price_a: pd.Series, price_b: pd.Series, beta: float) -> tuple[bool, str]:
    """Gate a pair BEFORE its spread is trusted. Runs on the formation window only.

    Cointegration alone is not enough of a filter. adf_pvalue() will happily score
    a spread built from a meaningless beta, and hedge_ratio() will happily fit a
    beta to two uncorrelated series. Both need to be screened out up front.

    @param beta  the hedge ratio already fitted on this same window
    @return (ok, reason). reason is "ok" when the pair passes.
    """
    if np.isnan(beta):
        return False, "beta is nan (degenerate fit)"

    corr = correlation(price_a, price_b)
    if np.isnan(corr):
        return False, "correlation is nan (insufficient overlap)"
    if abs(corr) < MIN_CORRELATION:
        return False, f"|corr| {abs(corr):.2f} < {MIN_CORRELATION} (legs do not move together)"

    if beta <= MIN_BETA:
        return False, f"beta {beta:.2f} <= {MIN_BETA} (would be long both legs, not a spread)"

    return True, "ok"


# Old hedge ratio: finds regression line through OLS.
# This minimizes distance of the line from price_a's plots
# So pairs A, B would give different results compared to B,A
# Used TLS to make this consistent -- then standardized TLS to fix its scale bias.

# def hedge_ratio(price_a, price_b):
#     b_with_const = sm.add_constant(price_b)      # allow an intercept
#     model = sm.OLS(price_a, b_with_const).fit()  # fit A ~= intercept + beta*B
#     return model.params.iloc[1]                   # return slope beta


# --- spread ------------------------------------------------------------------

# To find spread, if stock moves up for example, one is larger than the other 200 and 100
# current difference is 100
# If both move up by 100%, they become 400 and 200
# A - B = 200 / incorrect spread
# First normalize B, 200/100 = 2, so Let B = 200, both move up and becoome
# 400 and 400, 400 - 400 = 0 -> correct spread showing they didn't move apart more
def spread(price_a: pd.Series, price_b: pd.Series) -> pd.Series:
    """
    Calculates the hedge-ratio adjusted spread between two asset prices.
    Returns: pd.Series: The calculated spread (a - beta * b),
    or an empty Series if there are fewer than 100 overlapping days.
    """
    # align dates and handle Nan values, then pass clean data downstream
    combined = pd.concat([price_a, price_b], axis=1).dropna()
    if len(combined) < 100:      # not enough overlapping days -> return empty
        return pd.Series(dtype=float)
    a = combined.iloc[:, 0]
    b = combined.iloc[:, 1]
    beta = hedge_ratio(a, b)   # passing clean series
    return a - beta * b


# Builds the spread using a PROVIDED beta instead of fitting a fresh one.
# Needed for walk-forward: beta learned on the formation period is applied unchanged
# to the trading period, so trading data never leaks into pair fitting.
# Differs from spread() only in that it uses the beta passed in rather than refitting.
def spread_with_beta(price_a: pd.Series, price_b: pd.Series, beta: float) -> pd.Series:
    combined = pd.concat([price_a, price_b], axis=1).dropna()
    a = combined.iloc[:, 0]
    b = combined.iloc[:, 1]
    return a - beta * b


def adf_pvalue(spread_series):
    result = adfuller(spread_series.dropna())
    return result[1]   # index 1 is the p-value


def half_life(spread_series):
    clean = spread_series.dropna()
    lag = clean.shift(1).dropna()             # yesterday's spread
    delta = clean.diff().dropna()             # today's change
    lag = lag.loc[delta.index]                # align both
    theta = sm.OLS(delta, sm.add_constant(lag)).fit().params.iloc[1]
    return -np.log(2) / theta


# --- screening ---------------------------------------------------------------

def screen_all(close, pairs):
    """Score every candidate pair. Order of checks matters: correlation and beta
    sign are screened BEFORE the ADF p-value is trusted, because ADF returns a
    number for a nonsense spread just as readily as for a real one."""
    results = []
    rejected = []
    for sector, a, b in pairs:
        if a not in close.columns or b not in close.columns:
            print(f"  skipping {a}-{b}: missing data")
            continue

        combined = pd.concat([close[a], close[b]], axis=1).dropna()
        if len(combined) < 100:
            continue

        beta = hedge_ratio(close[a], close[b])
        ok, reason = pair_is_valid(close[a], close[b], beta)
        if not ok:
            rejected.append({"sector": sector, "stock_a": a, "stock_b": b,
                             "beta": round(beta, 3) if not np.isnan(beta) else None,
                             "reason": reason})
            continue

        s = spread_with_beta(close[a], close[b], beta)
        p = adf_pvalue(s)
        hl = half_life(s)
        if hl <= 0 or not np.isfinite(hl):   # not mean-reverting, skip
            continue
        results.append({
            "sector": sector,
            "stock_a": a,
            "stock_b": b,
            "corr": round(correlation(close[a], close[b]), 2),
            "beta": round(beta, 3),
            "adf_pvalue": round(p, 4),
            "half_life": round(hl, 1),
            "passes": p < COINT_THRESHOLD and 5 < hl < 60
        })

    if rejected:
        print(f"\n{len(rejected)} pairs rejected by pair_is_valid before ADF:")
        print(pd.DataFrame(rejected).to_string(index=False))

    return pd.DataFrame(results).sort_values("adf_pvalue").reset_index(drop=True)


if __name__ == "__main__":
    from data_layer import load_panel
    folder = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"
    close = load_panel(folder, tickers=["COALINDIA", "ONGC"])

    # Pass panda series of date, close to spread, and store a date, spread series in s
    s = spread(close["COALINDIA"], close["ONGC"])

    print("correlation:", round(correlation(close["COALINDIA"], close["ONGC"]), 3))
    print("beta:", round(hedge_ratio(close["COALINDIA"], close["ONGC"]), 3))
    print(s.describe())
    print("ADF p-value:", adf_pvalue(s))
    print("half-life:", half_life(s), "days")

    # plot both stocks spread and raw close prices
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    close[["COALINDIA", "ONGC"]].plot(ax=axes[0], title="COALINDIA vs ONGC - close prices")
    s.plot(ax=axes[1], title="COALINDIA - beta*ONGC spread", color="orange")

    plt.tight_layout()
    plt.show()   # one show -> both plots appear together, press Q to close
