import pandas as pd
import statsmodels.api as sm

# Scores the backtest result: Sharpe, drawdown, calmar ratio, and beta-to-index (market-neutrality check).
def sharpe(daily_returns: pd.Series) -> float:
    return daily_returns.mean() / daily_returns.std() * (252 ** 0.5)

def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()      # highest point reached so far
    drawdown = equity - running_max    # how far below that peak each day
    return drawdown.min()              # the deepest hole

def calmar(equity: pd.Series, margin: float) -> float:
    """
    Calmar ratio: annualized return divided by the max drawdown,
    Unlike Sharpe, it only cares about the single worst drawdown, not day-to-day
    wiggle, so a strategy with one nasty quarter gets punished here even if its
    daily volatility elsewhere looks fine.
    """
    ann_return = annualized_return(equity, margin)         # e.g. 0.24 = 24%/yr
    max_dd_fraction = abs(max_drawdown(equity)) / margin   # worst drawdown as a fraction of margin, same units as ann_return
    return ann_return / max_dd_fraction if max_dd_fraction != 0 else float("nan")  # guard: no drawdown yet (e.g. very short equity curve) -> undefined ratio, not a divide-by-zero crash

def beta_to_index(strat_returns: pd.Series, index_returns: pd.Series) -> float:
    combined = pd.concat([strat_returns, index_returns], axis=1).dropna()
    model = sm.OLS(combined.iloc[:, 0],
                   sm.add_constant(combined.iloc[:, 1])).fit()
    return model.params.iloc[1]        # slope = beta (should be ~0)

def annualized_return(equity: pd.Series, margin: float) -> float:
    years = len(equity) / 252
    total_return = equity.iloc[-1] / margin
    return (1 + total_return) ** (1 / years) - 1

def summary(result: pd.DataFrame, margin: float,
            index_returns: pd.Series = None) -> None:
    """Print all metrics for a backtest result in one call."""
    daily_ret = result["net_pnl"] / margin

    print(f"total net P&L      : {result['net_pnl'].sum():.2f}")
    print(f"annualized return  : {annualized_return(result['equity'], margin) * 100:.1f}%")
    print(f"annualized Sharpe  : {sharpe(daily_ret):.2f}")
    print(f"max drawdown       : {max_drawdown(result['equity']):.2f} pts")

    if index_returns is not None:
        b = beta_to_index(daily_ret, index_returns)
        print(f"beta to Nifty      : {b:.3f}  (~0 = market-neutral)")