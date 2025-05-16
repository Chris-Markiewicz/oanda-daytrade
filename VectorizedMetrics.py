# VectorizedMetrics.py
import pandas as pd
import numpy as np

def calculate_vectorized_pnl_and_returns(price_series: pd.Series, 
                                         signals: pd.Series, 
                                         initial_capital: float = 100000.0, # For equity curve
                                         spread_cost_pct: float = 0.0002,
                                         transaction_fee_pct: float = 0.0):
    """
    Calculates strategy log returns and a simple equity curve/total P&L.
    """
    if not isinstance(price_series, pd.Series) or not isinstance(signals, pd.Series):
        raise ValueError("price_series and signals must be pandas Series.")
    
    common_index = price_series.index.intersection(signals.index)
    price_series = price_series.loc[common_index]
    signals = signals.loc[common_index]
    if price_series.empty or len(price_series) < 2:
        return pd.Series(dtype=float), 0.0, pd.Series(dtype=float) # returns, pnl, equity

    asset_log_returns = np.log(price_series / price_series.shift(1))
    strategy_log_returns = asset_log_returns * signals.shift(1) # Signal from t-1 applies to return from t-1 to t
    strategy_log_returns.fillna(0, inplace=True)

    trades = signals.shift(1).diff().fillna(0).ne(0)
    total_cost_pct = spread_cost_pct + transaction_fee_pct
    transaction_costs = trades * total_cost_pct
    
    net_strategy_log_returns = strategy_log_returns - transaction_costs
    net_strategy_log_returns.fillna(0, inplace=True)

    # Calculate equity curve and total P&L
    # This is a simplified equity curve based on log returns.
    # For more accuracy with compounding, use simple returns, but log returns are fine for comparison.
    cumulative_log_returns = net_strategy_log_returns.cumsum()
    equity_curve = initial_capital * np.exp(cumulative_log_returns)
    equity_curve.iloc[0] = initial_capital # Start with initial capital
    
    total_pnl = equity_curve.iloc[-1] - initial_capital
    
    return net_strategy_log_returns, total_pnl, equity_curve


# calculate_sharpe_ratio and get_daily_returns_from_bar_returns remain the same
def calculate_sharpe_ratio(strategy_returns: pd.Series, 
                           annual_risk_free_rate: float = 0.0,
                           periods_per_year: int = 252):
    if not isinstance(strategy_returns, pd.Series):
        raise ValueError("strategy_returns must be a pandas Series.")
    if strategy_returns.empty or len(strategy_returns) < 20:
        return -np.inf
    mean_return = strategy_returns.mean()
    std_return = strategy_returns.std()
    if std_return == 0 or np.isnan(std_return) or std_return < 1e-9: # Added tolerance for near-zero std
        # If mean return is positive with zero std, it's technically infinite Sharpe.
        # Cap it or return a large number if desired, or handle as per strategy.
        # For now, if strictly zero std, and positive mean, could be an issue or perfect (unlikely).
        return -np.inf if mean_return <= (annual_risk_free_rate / periods_per_year) else np.inf # Or a large positive number
    sharpe_ratio = (mean_return - (annual_risk_free_rate / periods_per_year)) / std_return
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(periods_per_year)
    return annualized_sharpe_ratio if np.isfinite(annualized_sharpe_ratio) else -np.inf

def get_daily_returns_from_bar_returns(bar_returns: pd.Series):
    if not isinstance(bar_returns.index, pd.DatetimeIndex):
        raise ValueError("bar_returns must have a DatetimeIndex for daily resampling.")
    daily_log_returns = bar_returns.resample('D').sum()
    return daily_log_returns