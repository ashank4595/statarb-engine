# signal_execution_example.py
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
