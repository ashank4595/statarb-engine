# signal.py
# Turns a spread series into a z-score signal:
# how many std-devs from the rolling mean is the spread right now?
#
# Execution example with mock numbers: docs/zscore_signal_execution_example.md

import pandas as pd

# z > +2  -> short the spread (too wide, bet it narrows)
# z < -2  -> long the spread  (too narrow, bet it widens)
# |z|<0.5 -> exit             (back to normal, take profit)
# |z|>3.5 -> STOP_LOSS        (relationship may have broken permanently)

ENTRY_THRESHOLD = 2.0    # open a trade when |z| crosses this
EXIT_THRESHOLD  = 0.5    # close a trade when |z| falls below this
STOP_LOSS  = 3.5    # emergency exit if spread keeps running away


def zscore(spread_series: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute the rolling z-score of the spread series.

    Normalises the spread so every pair speaks the same language:
    z=2 means "2 standard deviations above the rolling mean" regardless
    of the pair's raw price scale.

    Args:
        spread_series: daily spread values (A - beta*B), one per date.
        window:        number of trading days for the rolling mean/std.
                       Default 60 (~3 months). First `window` rows are NaN
                       because there is not enough history to calibrate yet.

    Returns:
        pd.Series: z-scores indexed by date. NaN for the first `window` rows.
    """
    mean = spread_series.rolling(window).mean()   # rolling average of spread
    std  = spread_series.rolling(window).std()    # rolling std dev of spread
    return (spread_series - mean) / std           # how many stds from mean


def positions(zscore_series: pd.Series) -> pd.Series:
    """
    Convert daily z-scores into a position signal for each day.
    Iterates day by day (loop required because each day's position depends
    on yesterday's position, you hold a trade until exit/STOP_LOSS fires).

    Position values:
        +1.0  long the spread  (z crossed below -ENTRY_THRESHOLD)
        -1.0  short the spread (z crossed above +ENTRY_THRESHOLD)
         0.0  flat             (no position — exit or STOP_LOSS_LOSS triggered)

    Args:
        zscore_series: daily z-scores from zscore(), indexed by date.

    Returns:
        pd.Series: position signal indexed by date.
    """
    pos = pd.Series(0.0, index=zscore_series.index)

    for i in range(1, len(zscore_series)):
        z    = zscore_series.iloc[i]
        prev = pos.iloc[i - 1]   # what position were we in yesterday?

        if pd.isna(z):
            pos.iloc[i] = 0.0                  # no signal yet (still in warm-up)

        elif abs(z) > STOP_LOSS:
            pos.iloc[i] = 0.0                  # STOP_LOSS loss: spread ran too far

        elif prev != 0 and abs(z) < EXIT_THRESHOLD:
            pos.iloc[i] = 0.0                  # exit: spread reverted to normal

        elif z > ENTRY_THRESHOLD:
            pos.iloc[i] = -1.0                 # short: spread too wide

        elif z < -ENTRY_THRESHOLD:
            pos.iloc[i] =  1.0                 # long: spread too narrow

        else:
            pos.iloc[i] = prev                 # hold: no trigger fired, stay put

    return pos


if __name__ == "__main__":
    # Test on ONGC data
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_layer import load_panel
    from candidate_pairs.cointegration import spread
    import matplotlib.pyplot as plt

    folder = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"
    close = load_panel(folder, tickers=["NIFTY", "BANKNIFTY"])

    s = spread(close["NIFTY"], close["BANKNIFTY"])
    z = zscore(s)
    p = positions(z)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    s.plot(ax=axes[0], title="Spread (A - beta*B)", color="blue")

    z.plot(ax=axes[1], title="Z-score", color="orange")
    axes[1].axhline( ENTRY_THRESHOLD, color="red",   linestyle="--", alpha=0.7, label="entry +2")
    axes[1].axhline(-ENTRY_THRESHOLD, color="green", linestyle="--", alpha=0.7, label="entry -2")
    axes[1].axhline( EXIT_THRESHOLD,  color="gray",  linestyle=":",  alpha=0.5, label="exit +0.5")
    axes[1].axhline(-EXIT_THRESHOLD,  color="gray",  linestyle=":",  alpha=0.5, label="exit -0.5")
    axes[1].legend(fontsize=8)

    p.plot(ax=axes[2], title="Position  (+1=long spread, -1=short spread, 0=flat)", color="purple")
    axes[2].axhline(0, color="black", linestyle="-", alpha=0.3)
    # inspect what happened around the double trade
    window = p[(p.index >= "2025-01-01") & (p.index <= "2025-04-01")]
    changes = window[window.diff() != 0]   # days the position changed
    print(changes)

    # and the z-scores on those days
    print(z.loc[changes.index])
    plt.tight_layout()
    plt.show()
