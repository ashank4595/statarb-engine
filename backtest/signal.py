# signal.py
# Turns a spread series into a z-score signal:
# how many std-devs from the rolling mean is the spread right now?
#
# EXECUTION EXAMPLE (made-up numbers to trace the flow):
#
# 1. load_panel() returns a DataFrame (close):
#
#    date         COALINDIA    ONGC
#    2023-01-02   400.00       200.00
#    2023-01-03   402.00       201.00
#    2023-01-04   398.00       203.00
#
# 2. spread(close["COALINDIA"], close["ONGC"]) 
#       computes A - beta*B, compressing raw spread (A -B)
#    Say beta = 1.8 (from regression). Then returns:
#
#    date         spread
#    2023-01-02   400 - 1.8*200 =  40.00
#    2023-01-03   402 - 1.8*201 =  40.20
#    2023-01-04   398 - 1.8*203 = -7.40   <- raw spread compressed
#
# 3. zscore(spread, window=3) computes rolling mean and std over 3 days.
#    For the 3rd row (first row with enough history):
#    mean = (40.00 + 40.20 + (-7.40)) / 3 = 24.27
#    std  = std([40.00, 40.20, -7.40])     = 27.12
#    z = (x - μ) / σ
#      = (-7.40 - 24.27) / 27.12        = -1.17 
#
#    date         zscore
#    2023-01-02   NaN      <- not enough history yet
#    2023-01-03   NaN      <- not enough history yet
#    2023-01-04   -1.17    <- first valid z-score
#
# 4. positions(zscore) reads the z-score each day and decides what to hold.
#    z = -1.17 -> not past entry threshold of -2.0 -> stay flat (0)
#
#    date         position
#    2023-01-02   0         <- flat (no signal yet)
#    2023-01-03   0         <- flat (no signal yet)
#    2023-01-04   0         <- flat (z = -1.17, not past -2.0 threshold)
#
#    If on day 5 z dropped to -2.3:
#    position = +1.0        <- long the spread (bet it widens back to mean)
#
#    If on day 6 |z| fell back to 0.3 (< EXIT_THRESHOLD of 0.5):
#    position = 0.0         <- exit, spread reverted to normal, take profit

import pandas as pd

# z > +2  -> short the spread (too wide, bet it narrows)
# z < -2  -> long the spread  (too narrow, bet it widens)
# |z|<0.5 -> exit             (back to normal, take profit)
# |z|>3.5 -> stop loss        (relationship may have broken permanently)

ENTRY_THRESHOLD = 2.0    # open a trade when |z| crosses this
EXIT_THRESHOLD  = 0.5    # close a trade when |z| falls below this
STOP_THRESHOLD  = 3.5    # emergency exit if spread keeps running away


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

    Example:
        spread = pd.Series([40.0, 40.2, -7.4])
        zscore(spread, window=3)
        # -> [NaN, NaN, -1.17]
        # -1.17 means the spread is 1.17 std-devs below its 3-day mean.
    """
    mean = spread_series.rolling(window).mean()   # rolling average of spread
    std  = spread_series.rolling(window).std()    # rolling std dev of spread
    return (spread_series - mean) / std           # how many stds from mean


def positions(zscore_series: pd.Series) -> pd.Series:
    """
    Convert daily z-scores into a position signal for each day.

    Iterates day by day (loop required because each day's position depends
    on yesterday's position — you hold a trade until exit/stop fires).

    Position values:
        +1.0  long the spread  (z crossed below -ENTRY_THRESHOLD)
        -1.0  short the spread (z crossed above +ENTRY_THRESHOLD)
         0.0  flat             (no position — exit or stop triggered)

    Args:
        zscore_series: daily z-scores from zscore(), indexed by date.

    Returns:
        pd.Series: position signal indexed by date.

    Example:
        z = pd.Series([NaN, NaN, -1.17, -2.30, -0.30])
        positions(z)
        # day 0-1: NaN z -> flat (0)
        # day 2:   z=-1.17 -> not past -2.0 -> flat (0)
        # day 3:   z=-2.30 -> past -2.0 -> long (+1)
        # day 4:   z=-0.30 -> |z|<0.5, was long -> EXIT (0)
    """
    pos = pd.Series(0.0, index=zscore_series.index)

    for i in range(1, len(zscore_series)):
        z    = zscore_series.iloc[i]
        prev = pos.iloc[i - 1]   # what position were we in yesterday?

        if pd.isna(z):
            pos.iloc[i] = 0.0                  # no signal yet (still in warm-up)

        elif abs(z) > STOP_THRESHOLD:
            pos.iloc[i] = 0.0                  # stop loss: spread ran too far

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
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_layer import load_panel
    from candidate_pairs.cointegration import spread
    import matplotlib.pyplot as plt

    folder = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"
    close = load_panel(folder, tickers=["COALINDIA", "ONGC"])
    # close is now a DataFrame:
    # date         COALINDIA    ONGC
    # 2023-01-02   400.10       185.20
    # 2023-01-03   402.50       186.10
    # ...          ...          ...

    s = spread(close["COALINDIA"], close["ONGC"])
    # s is a Series: date -> spread value (COALINDIA - beta*ONGC)
    # date
    # 2023-01-02    40.10
    # 2023-01-03    39.80
    # ...

    z = zscore(s)
    # z is a Series: date -> z-score (NaN for first 60 rows)
    # date
    # 2023-01-02    NaN
    # ...
    # 2023-04-20    -1.82
    # 2023-04-21    -2.14   <- crosses -2.0, trade opens next day

    p = positions(z)
    # p is a Series: date -> position (+1, -1, or 0)
    # date
    # 2023-04-20     0.0    <- z=-1.82, not past threshold yet
    # 2023-04-21    +1.0    <- z=-2.14, go long the spread
    # ...
    # 2023-05-10     0.0    <- |z| dropped below 0.5, exit

    # plot all three on one figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    s.plot(ax=axes[0], title="Spread (COALINDIA - beta*ONGC)", color="blue")

    z.plot(ax=axes[1], title="Z-score", color="orange")
    axes[1].axhline( ENTRY_THRESHOLD, color="red",   linestyle="--", alpha=0.7, label="entry +2")
    axes[1].axhline(-ENTRY_THRESHOLD, color="green", linestyle="--", alpha=0.7, label="entry -2")
    axes[1].axhline( EXIT_THRESHOLD,  color="gray",  linestyle=":",  alpha=0.5, label="exit +0.5")
    axes[1].axhline(-EXIT_THRESHOLD,  color="gray",  linestyle=":",  alpha=0.5, label="exit -0.5")
    axes[1].legend(fontsize=8)

    p.plot(ax=axes[2], title="Position  (+1=long spread, -1=short spread, 0=flat)", color="purple")
    axes[2].axhline(0, color="black", linestyle="-", alpha=0.3)

    plt.tight_layout()
    plt.show()
