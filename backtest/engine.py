# Runs a signal through history without peeking at the future (t+1 rule) and applies costs to produce a P&L.
# engine.py
# Replays position signals through history to produce an honest P&L.
# The one rule: a position decided at day t's close earns day t+1's return.
# Execution example: docs/signal_execution_example.md (t+1 section)

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


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_layer import load_panel
    from candidate_pairs.cointegration import spread
    from backtest.zscore_signal import zscore, positions
    import matplotlib.pyplot as plt
    import numpy as np

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

    print(result.tail())
    print(f"\ntotal gross P&L : {result['gross_pnl'].sum():.2f}")
    print(f"total costs     : {result['costs'].sum():.2f}")
    print(f"total net P&L   : {result['net_pnl'].sum():.2f}")

    # --- convert to percent returns on margin capital ---
    # 1 unit of spread = long 1 STOCK_A + short beta STOCK_B
    # approximate margin to hold both legs (~20% of combined notional)
    notional = close[STOCK_A].mean() * 2               # rough: both legs similar size
    margin = notional * 0.20                           # ~20% margin requirement

    daily_ret = result["net_pnl"] / margin             # daily % return on capital

    years = len(daily_ret) / 252                       # trading days -> years
    total_return = result["equity"].iloc[-1] / margin
    annual_return = (1 + total_return) ** (1 / years) - 1

    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)

    print(f"\nmargin capital assumed : {margin:.0f}")
    print(f"total return           : {total_return * 100:.1f}%")
    print(f"annualized return      : {annual_return * 100:.1f}%")
    print(f"annualized Sharpe      : {sharpe:.2f}")

    result["equity"].plot(title=f"{STOCK_A}-{STOCK_B} pairs strategy: equity curve",
                          figsize=(12, 5))
    plt.axhline(0, color="black", alpha=0.3)
    plt.show()
