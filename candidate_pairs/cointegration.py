# Must run this as a module:
# statarb-engine % source venv/bin/activate
# (venv) statarb-engine % python3 -m candidate_pairs.cointegration

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



#draw the best-fit line through the A-vs-B dots, and return its slope (β)
def hedge_ratio(price_a, price_b):
    b_with_const = sm.add_constant(price_b)      # allow an intercept
    # Find Ordinary line of Least Squares
    model = sm.OLS(price_a, b_with_const).fit()  # fit A ≈ intercept + β·B
    return model.params.iloc[1]                   # return slope β

# To find spread, if stock moves up for example, one is larger than the other 200 and 100
# current difference is 100
# If both move up by 100%, they become 400 and 200
# A - B = 200 / incorrect spread
# First normalize B, 200/100 = 2, so Let B = 200, both move up and becoome 
# 400 and 400, 400 - 400 = 0 -> correct spread showing they didn't move apart more
def spread(price_a, price_b):
    # align dates and handle Nan values, then pass clean data downstream
    combined = pd.concat([price_a, price_b], axis=1).dropna()
    a = combined.iloc[:, 0]
    b = combined.iloc[:, 1]

    beta = hedge_ratio(a, b)   # passing clean series
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

if __name__ == "__main__":
    from data_layer import load_panel
    folder = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"
    close = load_panel(folder, tickers=["COALINDIA", "ONGC"])
    print(close.columns.tolist())


    # Pass panda series of date, close to spread, and store a date, spread series in s
    s = spread(close["COALINDIA"], close["ONGC"]) 
    
    print(s.describe())        # summary stats of the spread

    print("ADF p-value:", adf_pvalue(s))
    print("half-life:", half_life(s), "days")

    # plot both stocks spread and raw close prices
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    close[["COALINDIA", "ONGC"]].plot(ax=axes[0], title="COALINDIA vs ONGC - close prices")
    s.plot(ax=axes[1], title="COALINDIA - beta*ONGC spread", color="orange")

    plt.tight_layout()
    plt.show()   # one show → both plots appear together, press Q to close

