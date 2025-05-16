# VectorizedPerformanceCalculator.py

import pandas as pd
import numpy as np
from VectorizedMetrics import calculate_vectorized_pnl_and_returns, calculate_sharpe_ratio, get_daily_returns_from_bar_returns

class VectorizedPerformanceCalculator:
    def __init__(self, symbol: str, ml_model=None,
                 initial_capital: float = 100000.0,
                 annual_risk_free_rate: float = 0.0,
                 spread_cost_pct: float = 0.0002,
                 transaction_fee_pct: float = 0.0001,
                 confidence_threshold: float = 0.0, # 0.0 means disabled
                 use_trend_filter: bool = False,
                 trend_filter_ma_period: int = 50,
                 price_col_for_trend_filter: str = 'price'): # Column for trend MA
        self.symbol = symbol
        self.ml_model = ml_model
        self.initial_capital = initial_capital
        self.annual_risk_free_rate = annual_risk_free_rate
        self.spread_cost_pct = spread_cost_pct
        self.transaction_fee_pct = transaction_fee_pct
        
        self.confidence_threshold = confidence_threshold
        self.use_trend_filter = use_trend_filter
        self.trend_filter_ma_period = trend_filter_ma_period
        self.price_col_for_trend_filter = price_col_for_trend_filter # Usually 'price'


    def calculate_performance_metrics(self,
                                      price_and_features_df: pd.DataFrame, # Must contain price_col_for_trend_filter and features
                                      feature_columns: list = None,      # Features for ml_model.predict
                                      # signals_series is removed, as we always generate from ml_model here
                                      price_column_name_for_pnl: str = 'price'): # Price used for P&L calc
        
        default_metrics = {"pnl": 0.0, "sharpe": -np.inf, "equity_curve": pd.Series(dtype=float), "final_signals": pd.Series(dtype=int)}

        if self.ml_model is None: 
            print("VPC Error: No ML model provided."); return default_metrics
        if feature_columns is None or not all(col in price_and_features_df.columns for col in feature_columns):
            print(f"VPC Error: Missing or invalid feature_columns. Need: {feature_columns}, Have: {price_and_features_df.columns}"); return default_metrics
        
        X_features = price_and_features_df[feature_columns]
        if X_features.empty: 
            print("VPC Error: X_features is empty."); return default_metrics

        # --- 1. Get Raw Model Output (Predictions or Probabilities) ---
        raw_signals = pd.Series(dtype=int, index=X_features.index) # Initialize

        try:
            if self.confidence_threshold > 0.0 and hasattr(self.ml_model, 'predict_proba'):
                probabilities = self.ml_model.predict_proba(X_features) # Array of [P(class0), P(class1), ...]
                
                # Determine index for class 1 (UP) and class -1 (DOWN)
                # This assumes model.classes_ exists and is ordered, e.g. [-1, 1]
                idx_neg, idx_pos = -1, -1
                try:
                    if -1 in self.ml_model.classes_: idx_neg = list(self.ml_model.classes_).index(-1)
                    if 1 in self.ml_model.classes_: idx_pos = list(self.ml_model.classes_).index(1)
                except AttributeError: # If model.classes_ doesn't exist, assume standard order from training
                    print("Warning: model.classes_ not found. Assuming prob order [P(-1), P(1)] if binary.")
                    if probabilities.shape[1] == 2: # Common for binary
                        idx_neg, idx_pos = 0, 1 
                    else: # Cannot determine probability order
                         print("Error: Cannot determine probability order for confidence thresholding.")
                         return default_metrics


                prob_down = probabilities[:, idx_neg] if idx_neg != -1 else np.zeros(len(probabilities))
                prob_up = probabilities[:, idx_pos] if idx_pos != -1 else np.zeros(len(probabilities))

                long_condition = (prob_up > self.confidence_threshold) & (prob_up > prob_down)
                short_condition = (prob_down > self.confidence_threshold) & (prob_down > prob_up)
                
                raw_signals[long_condition] = 1
                raw_signals[short_condition] = -1
                raw_signals.fillna(0, inplace=True) # Neutral if no condition met
            else:
                predictions = self.ml_model.predict(X_features)
                raw_signals = pd.Series(predictions.astype(int), index=X_features.index)
        except Exception as e:
            print(f"VPC Error during model prediction/probability: {e}"); return default_metrics

        # --- 2. Apply Trend Filter (if enabled) ---
        final_trade_signals = raw_signals.copy()
        if self.use_trend_filter:
            if self.price_col_for_trend_filter not in price_and_features_df.columns:
                print(f"VPC Error: Trend filter price column '{self.price_col_for_trend_filter}' not found."); return default_metrics
            
            trend_ma = price_and_features_df[self.price_col_for_trend_filter].ewm(span=self.trend_filter_ma_period, adjust=False).mean()
            
            # Align trend_ma with signals index for safe comparison
            trend_ma = trend_ma.reindex(final_trade_signals.index) # Ensure alignment

            # Filter long signals: only if price > trend_ma
            long_override_condition = (final_trade_signals == 1) & (price_and_features_df[self.price_col_for_trend_filter].loc[final_trade_signals.index] < trend_ma)
            final_trade_signals[long_override_condition] = 0
            
            # Filter short signals: only if price < trend_ma
            short_override_condition = (final_trade_signals == -1) & (price_and_features_df[self.price_col_for_trend_filter].loc[final_trade_signals.index] > trend_ma)
            final_trade_signals[short_override_condition] = 0

        # --- 3. Calculate P&L and Sharpe using final_trade_signals ---
        price_series_for_pnl = price_and_features_df[price_column_name_for_pnl]
        common_index = price_series_for_pnl.index.intersection(final_trade_signals.index)
        
        aligned_price_data = price_series_for_pnl.loc[common_index]
        aligned_final_signals = final_trade_signals.loc[common_index]

        if aligned_price_data.empty or len(aligned_price_data) < 2:
            return default_metrics

        bar_strategy_log_returns, total_pnl, equity_curve = calculate_vectorized_pnl_and_returns(
            price_series=aligned_price_data,
            signals=aligned_final_signals,
            initial_capital=self.initial_capital,
            spread_cost_pct=self.spread_cost_pct,
            transaction_fee_pct=self.transaction_fee_pct
        )

        sharpe_ratio = -np.inf
        if not bar_strategy_log_returns.empty:
            try:
                daily_strategy_log_returns = get_daily_returns_from_bar_returns(bar_strategy_log_returns)
                if not daily_strategy_log_returns.empty and len(daily_strategy_log_returns) >= 20:
                    sharpe_ratio = calculate_sharpe_ratio(
                        strategy_returns=daily_strategy_log_returns,
                        annual_risk_free_rate=self.annual_risk_free_rate
                    )
            except ValueError: # If index is not datetime
                pass # Sharpe remains -inf
        
        return {"pnl": total_pnl, "sharpe": sharpe_ratio, "equity_curve": equity_curve, "final_signals": aligned_final_signals}