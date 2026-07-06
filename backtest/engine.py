# Runs a signal through history without peeking at the future (t+1 rule) and applies costs to produce a P&L.
# engine.py
# Replays position signals through history to produce an honest P&L.
# The one rule: a position decided at day t's close earns day t+1's return.

import pandas as pd


def backtest_pair(spread_series: pd.Series,
                  positions: pd.Series,
                  cost_per_unit: float = 0.05) -> pd.DataFrame:
    """
    Compute daily P&L for one pair's position signal, look-ahead-safe.

    The t+1 rule: positions.shift(1) ensures a position decided using
    day t's close only earns day t+1's spread change. Without this shift
    the backtest silently earns the very move that triggered its signal.

    Args:
        spread_series: daily spread values (A - beta*B) from cointegration.spread.
        positions:     daily position signal (+1/-1/0) from zscore_signal.positions.
        cost_per_unit: cost of trading 1 unit of the spread, in spread points.
                       Charged on every position change (entry, exit, flip).

    Returns:
        pd.DataFrame indexed by date with columns:
            gross_pnl  - daily P&L before costs
            costs      - transaction costs paid that day
            net_pnl    - gross_pnl minus costs
            equity     - cumulative net P&L (the equity curve)
    """
    spread_returns = spread_series.diff()            # daily change in spread

    gross_pnl = positions.shift(1) * spread_returns  # t+1 rule: yesterday's
                                                     # position earns today's move

    trades = positions.diff().abs()                  # units traded on each change
    costs = trades * cost_per_unit                   # cost of those trades

    net_pnl = gross_pnl - costs

    result = pd.DataFrame({
        "gross_pnl": gross_pnl,
        "costs": costs,
        "net_pnl": net_pnl,
    }).fillna(0.0)

    result["equity"] = result["net_pnl"].cumsum()    # running total = equity curve
    return result


def backtest_pair_loop(spread_series: pd.Series,
                       positions: pd.Series,
                       cost_per_unit: float = 0.05) -> pd.DataFrame:
    """
    Same as backtest_pair() but written as an explicit day-by-day loop
    instead of vectorized. Produces identical results — used to prove
    that "hold position, earn daily change" equals tracking trades.

    The line `yesterday_position = positions.iloc[i - 1]` IS the t+1 rule:
    reach back one day for the position, pair it with today's price change.
    That is exactly what positions.shift(1) does in the vectorized version.

    Args:
        spread_series: daily spread values (A - beta*B).
        positions:     daily position signal (+1/-1/0).
        cost_per_unit: cost of trading 1 unit of the spread.

    Returns:
        pd.DataFrame with gross_pnl, costs, net_pnl, equity (same as backtest_pair).
    """
    dates = spread_series.index
    gross_pnl = []
    costs = []

    for i in range(len(dates)):
        if i == 0:
            # first day: no previous position, no previous price -> no P&L
            gross_pnl.append(0.0)
            costs.append(0.0)
            continue

        # position decided YESTERDAY earns the move from yesterday to today
        yesterday_position = positions.iloc[i - 1]
        spread_change = spread_series.iloc[i] - spread_series.iloc[i - 1]
        gross_pnl.append(yesterday_position * spread_change)

        # cost: did the position change from yesterday to today?
        position_change = abs(positions.iloc[i] - positions.iloc[i - 1])
        costs.append(position_change * cost_per_unit)

    result = pd.DataFrame({
        "gross_pnl": gross_pnl,
        "costs": costs,
    }, index=dates)
    result["net_pnl"] = result["gross_pnl"] - result["costs"]
    result["equity"] = result["net_pnl"].cumsum()
    return result


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_layer import load_panel, load_daily_close
    from candidate_pairs.cointegration import spread
    from backtest.zscore_signal import zscore, positions
    from backtest.evaluate import summary
    import matplotlib.pyplot as plt

    # --- pick the pair to backtest (swap these two lines to test another pair) ---
    # top passing pairs from pair_screen_results.csv:
    #   COALINDIA-ONGC   (p=0.0005, hl=14)
    #   BALKRISIND-EXIDEIND (p=0.0014, hl=17)
    #   FEDERALBNK-CUB   (p=0.0016, hl=17)
    #   HAVELLS-CROMPTON (p=0.0017, hl=26)
    #   SUNPHARMA-LUPIN  (p=0.0024, hl=20)
    STOCK_A = "HAVELLS"
    STOCK_B = "CROMPTON"

    folder = "/Users/ashankawasthy/Desktop/quant_trading/derived_data/futures"
    close = load_panel(folder, tickers=[STOCK_A, STOCK_B])

    s = spread(close[STOCK_A], close[STOCK_B])
    z = zscore(s)
    p = positions(z)

    result = backtest_pair(s, p)

    # margin capital: ~20% of combined notional of both legs
    notional = close[STOCK_A].mean() * 2
    margin = notional * 0.20

    # load Nifty index to check market-neutrality (beta should be ~0)
    nifty = load_daily_close(os.path.join(folder, "NIFTY_-I.csv"))
    nifty_returns = nifty.pct_change()

    print(f"=== {STOCK_A}-{STOCK_B} pairs strategy ===")
    summary(result, margin, index_returns=nifty_returns)

    result["equity"].plot(title=f"{STOCK_A}-{STOCK_B} pairs strategy: equity curve",
                          figsize=(12, 5))
    plt.axhline(0, color="black", alpha=0.3)
    plt.show()
