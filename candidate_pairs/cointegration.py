# Must run this as a module:
# statarb-engine % source venv/bin/activate
# (venv) statarb-engine % python3 -m candidate_pairs.cointegration

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import pandas as pd


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

if __name__ == "__main__":
    from data_layer import load_panel
    folder = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"
    close = load_panel(folder, tickers=["AXISBANK", "SBIN"])
    print(close.columns.tolist())


    # Pass panda series of date, close to spread, and store a date, spread series in s
    s = spread(close["AXISBANK"], close["SBIN"]) 
    print(s.describe())        # summary stats of the spread

    print("ADF p-value:", adf_pvalue(s))

    s.plot(title="COALINDIA - beta*ONGC spread")
    import matplotlib.pyplot as plt
    plt.show()
