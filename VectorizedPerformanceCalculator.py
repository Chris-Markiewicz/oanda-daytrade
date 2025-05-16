# VectorizedPerformanceCalculator.py

import pandas as pd
import numpy as np
from VectorizedMetrics import calculate_vectorized_pnl_and_returns, calculate_sharpe_ratio, get_daily_returns_from_bar_returns

class VectorizedPerformanceCalculator:
    def __init__(self, symbol: str, ml_model=None,
                 initial_capital: float = 100000.0,
                 annual_risk_free_rate: float = 0.0,
                 spread_cost_pct: float = 0.0002,
                 transaction_fee_pct: float = 0.0001):
        self.symbol = symbol
        self.ml_model = ml_model
        self.initial_capital = initial_capital
        self.annual_risk_free_rate = annual_risk_free_rate
        self.spread_cost_pct = spread_cost_pct
        self.transaction_fee_pct = transaction_fee_pct

    def calculate_performance_metrics(self,
                                      price_and_features_df: pd.DataFrame,
                                      feature_columns: list = None,
                                      signals_series: pd.Series = None,
                                      price_column_name: str = 'price'):
        
        default_metrics = {"pnl": 0.0, "sharpe": -np.inf, "equity_curve": pd.Series(dtype=float)}

        if signals_series is None:
            if self.ml_model is None: return default_metrics
            if feature_columns is None or not all(col in price_and_features_df.columns for col in feature_columns):
                return default_metrics
            X_features = price_and_features_df[feature_columns]
            if X_features.empty: return default_metrics
            try:
                model_predictions = self.ml_model.predict(X_features)
                signals = pd.Series(model_predictions, index=X_features.index)
            except Exception: return default_metrics
        else:
            signals = signals_series.copy()

        common_index = price_and_features_df[price_column_name].index.intersection(signals.index)
        aligned_price_data = price_and_features_df.loc[common_index, price_column_name]
        aligned_signals = signals.loc[common_index]

        if aligned_price_data.empty or len(aligned_price_data) < 2:
            return default_metrics

        bar_strategy_log_returns, total_pnl, equity_curve = calculate_vectorized_pnl_and_returns(
            price_series=aligned_price_data,
            signals=aligned_signals,
            initial_capital=self.initial_capital,
            spread_cost_pct=self.spread_cost_pct,
            transaction_fee_pct=self.transaction_fee_pct
        )

        if bar_strategy_log_returns.empty:
            # Return equity curve starting with initial capital if PnL could start
            return {"pnl": total_pnl, "sharpe": -np.inf, "equity_curve": equity_curve if not equity_curve.empty else pd.Series([self.initial_capital], index=[aligned_price_data.index[0]] if not aligned_price_data.empty else None)}


        try:
            daily_strategy_log_returns = get_daily_returns_from_bar_returns(bar_strategy_log_returns)
        except ValueError:
            return {"pnl": total_pnl, "sharpe": -np.inf, "equity_curve": equity_curve}
        
        if daily_strategy_log_returns.empty or len(daily_strategy_log_returns) < 20:
             sharpe_ratio = -np.inf
        else:
            sharpe_ratio = calculate_sharpe_ratio(
                strategy_returns=daily_strategy_log_returns,
                annual_risk_free_rate=self.annual_risk_free_rate
            )
        
        return {"pnl": total_pnl, "sharpe": sharpe_ratio, "equity_curve": equity_curve}