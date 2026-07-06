# Must run this as a module:
# statarb-engine % source venv/bin/activate
# (venv) statarb-engine % python3 -m candidate_pairs.cointegration

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# TLS (Total Least Squares) hedge ratio via the first principal component.
# Unlike OLS, this is symmetric: hedge_ratio(A, B) and hedge_ratio(B, A) give
# the same relationship, because it minimizes perpendicular distance to the
# line (not vertical), so the arbitrary choice of which stock is A disappears.
# def hedge_ratio(price_a, price_b):
#     combined = pd.concat([price_a, price_b], axis=1).dropna()
#     a = combined.iloc[:, 0]
#     b = combined.iloc[:, 1]

#     # covariance matrix of the two price series
#     cov = np.cov(a, b)

#     # eigenvectors of the covariance matrix; the one with the largest
#     # eigenvalue points along the "long axis" of the price cloud
#     eigenvalues, eigenvectors = np.linalg.eig(cov)
#     principal = eigenvectors[:, np.argmax(eigenvalues)]

#     # slope of that principal axis = the symmetric hedge ratio
#     return principal[1] / principal[0]

def hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """
    TLS/PCA hedge ratio for:
        spread = A - beta * B

    Returns beta such that A ≈ beta * B.
    """
    combined = pd.concat([price_a, price_b], axis=1).dropna()

    if len(combined) < 2:
        return np.nan

    a = combined.iloc[:, 0].to_numpy(dtype=float)
    b = combined.iloc[:, 1].to_numpy(dtype=float)

    cov = np.cov(a, b)

    # eigh is specifically for real symmetric covariance matrices.
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Direction of greatest shared movement.
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    # principal[0] = A-direction, principal[1] = B-direction.
    # Since spread is A - beta*B, beta should be dA/dB.
    if np.isclose(principal[1], 0):
        return np.nan

    return float(principal[0] / principal[1])

# Old hedge ratio: finds regression line through OLS.
# This minimizes distance of the line from price_a's plots
# So pairs A, B would give different results compared to B,A
# Used TLS to make this consistent

# #draw the best-fit line through the A-vs-B plots, and return its slope (β)
# def hedge_ratio(price_a, price_b):
#     b_with_const = sm.add_constant(price_b)      # allow an intercept
#     # Find Ordinary line of Least Squares
#     model = sm.OLS(price_a, b_with_const).fit()  # fit A c≈ intercept + β·B

#     return model.params.iloc[1]                   # return slope β

# To find spread, if stock moves up for example, one is larger than the other 200 and 100
# current difference is 100
# If both move up by 100%, they become 400 and 200
# A - B = 200 / incorrect spread
# First normalize B, 200/100 = 2, so Let B = 200, both move up and becoome 
# 400 and 400, 400 - 400 = 0 -> correct spread showing they didn't move apart more
def spread(price_a : pd.Series, price_b : pd.Series) -> pd.Series:
    """
    Calculates the hedge-ratio adjusted spread between two asset prices.
    Returns: pd.Series: The calculated spread (a - beta * b), 
    or an empty Series if there are fewer than 100 overlapping days.
    """
    # align dates and handle Nan values, then pass clean data downstream
    combined = pd.concat([price_a, price_b], axis=1).dropna()
    if len(combined) < 100:      # not enough overlapping days → return empty
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

def screen_all(close, pairs):
    results = []
    for sector, a, b in pairs:
        if a not in close.columns or b not in close.columns:
            print(f"  skipping {a}-{b}: missing data")
            continue
        s = spread(close[a], close[b])
        if len(s) < 100:   # empty or too short — spread() returned early
            continue
        p = adf_pvalue(s)
        hl = half_life(s)
        if hl <= 0 or not np.isfinite(hl):   # not mean-reverting, skip
            continue
        results.append({
            "sector": sector,
            "stock_a": a,
            "stock_b": b,
            "adf_pvalue": round(p, 4),
            "half_life": round(hl, 1),
            "passes": p < 0.05 and 5 < hl < 60
        })
    return pd.DataFrame(results).sort_values("adf_pvalue").reset_index(drop=True)

if __name__ == "__main__":
    from data_layer import load_panel
    folder = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"
    close = load_panel(folder, tickers=["COALINDIA", "ONGC"])

    # Pass panda series of date, close to spread, and store a date, spread series in s
    s = spread(close["COALINDIA"], close["ONGC"])

    print(s.describe())
    print("ADF p-value:", adf_pvalue(s))
    print("half-life:", half_life(s), "days")

    # plot both stocks spread and raw close prices
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    close[["COALINDIA", "ONGC"]].plot(ax=axes[0], title="COALINDIA vs ONGC - close prices")
    s.plot(ax=axes[1], title="COALINDIA - beta*ONGC spread", color="orange")

    plt.tight_layout()
    plt.show()   # one show → both plots appear together, press Q to close
