# ModelTrainer.py

import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta, timezone
import pickle

# Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# Sklearn utilities
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler # Import StandardScaler

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Custom modules
from BaseIndicator import BaseIndicator
from MovingAverages import MovingAverageFeature, MACrossover
from MomentumIndicators import LaggedReturnsFeature, RSIFeature, MACDFeature
from VolatilityIndicators import ATRFeature, BollingerBandsFeatures, VolatilitySTDFeature 
from TimeFeatures import HourFeature, DayOfWeekFeature 

from VectorizedMetrics import calculate_vectorized_pnl_and_returns, calculate_sharpe_ratio, get_daily_returns_from_bar_returns
from VectorizedPerformanceCalculator import VectorizedPerformanceCalculator


# --- Configuration Parameters ---
CONF_FILE = "oanda.cfg"
INSTRUMENT = "EUR_USD"
MODEL_FILENAME_PATTERN = f"ml_trader_model_TYPE_Scaled_VecSharpe_{INSTRUMENT.replace('_','')}_1min_5lags_trained.pkl" # Added _Scaled

CONFIDENCE_THRESHOLD_BACKTEST = 0.60 # Example: 0.0 to disable, 0.6 for 60% confidence
USE_TREND_FILTER_BACKTEST = True     # Example
TREND_FILTER_MA_PERIOD_BACKTEST = 50 # Example

# Data Fetching Parameters
DAYS_OF_DATA_TO_FETCH = 200 
END_OFFSET_DAYS = 1        
GRANULARITY_FETCH = "S5"   
BAR_LENGTH_MODEL = "1min"  

# Model & Feature Parameters
LAGS = 5
CV_SPLITS_FOR_TUNING = 3   

# Cost Parameters for VectorizedPerformanceCalculator
SPREAD_COST_PCT = 0.0001 
TRANSACTION_FEE_PCT = 0.00005 

# --- Helper Functions ---
def fetch_and_prepare_data(conf_file, instrument, days_to_fetch, end_offset_days, 
                           granularity_fetch, bar_length_model, 
                           indicators_to_apply: list = None):
    print(f"Initializing OANDA connection using {conf_file}...")
    api = tpqoa.tpqoa(conf_file)
    print(f"Fetching {days_to_fetch} days of {granularity_fetch} data for {instrument}, ending {end_offset_days} day(s) ago.")

    # Time Range Calc
    end_date = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=end_offset_days)
    end_date = end_date - timedelta(days= end_date.day, minutes= end_date.minute, seconds= end_date.second, microseconds=end_date.microsecond)
    start_date = end_date - timedelta(days=days_to_fetch)


    print(f"Calculated fetch period: Start={start_date.isoformat()}, End={end_date.isoformat()}")
    
    try:
        raw_hist_df = api.get_history(instrument=instrument, start=start_date, end=end_date, 
                                   granularity=granularity_fetch, price="M", localize=False)
    except Exception as e:
        print(f"Error fetching data from OANDA: {e}"); return None, []
    if raw_hist_df.empty: print("No data fetched."); return None, []
    print(f"Fetched {len(raw_hist_df)} data points at {granularity_fetch} granularity.")

    ohlc_dict = {'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'volume': 'sum'}
    resampled_data = raw_hist_df.resample(pd.to_timedelta(bar_length_model), label="right").agg(ohlc_dict)
    resampled_data.rename(columns={'c': 'price', 'o':'open', 'h':'high', 'l':'low'}, inplace=True)
    resampled_data.dropna(subset=['price', 'open', 'high', 'low'], inplace=True)

    if resampled_data.empty: print(f"No data after resampling to {bar_length_model}."); return None, []
    print(f"Resampled to {len(resampled_data)} {bar_length_model} bars.")

    df_with_features = resampled_data.copy()
    all_feature_names = []

    if indicators_to_apply is None:
        print("No indicators specified. Calculating base returns if 'price' column exists.")
        if 'price' in df_with_features.columns:
             df_with_features['returns'] = np.log(df_with_features['price'] / df_with_features['price'].shift(1))
        else:
            print("Warning: 'price' column not found, cannot calculate base 'returns'.")
    else:
        print("\nApplying indicators for feature engineering...")
        # Ensure 'returns' is calculated first if any indicator depends on it AND price_col is used for that indicator
        if 'price' in df_with_features.columns:
            df_with_features['returns'] = np.log(df_with_features['price'] / df_with_features['price'].shift(1))
        else:
             print("Warning: 'price' column not found, base 'returns' for dependent indicators might be missing.")

        for indicator_instance in indicators_to_apply:
            if isinstance(indicator_instance, BaseIndicator):
                print(f"  Calculating {indicator_instance.name}...")
                try:
                    # Some indicators might modify df_with_features by adding 'returns' if not present and needed
                    df_with_features = indicator_instance.calculate(df_with_features)
                    all_feature_names.extend(indicator_instance.get_feature_names())
                except Exception as e:
                    print(f"    Error calculating {indicator_instance.name}: {e}")
            else:
                print(f"  Warning: Item {indicator_instance} is not a valid BaseIndicator instance. Skipping.")
    
    all_feature_names = sorted(list(set(all_feature_names))) 
    
    if 'returns' not in df_with_features.columns and 'price' in df_with_features.columns:
        df_with_features['returns'] = np.log(df_with_features['price'] / df_with_features['price'].shift(1))
    elif 'returns' not in df_with_features.columns:
        print("CRITICAL: 'returns' column missing and cannot be calculated. Target variable cannot be generated.")
        return None, []

    df_with_features['direction'] = np.where(df_with_features['returns'].shift(-1) > 0, 1, -1)
    df_with_features.dropna(inplace=True)
    
    valid_feature_names = [col for col in all_feature_names if col in df_with_features.columns and not df_with_features[col].isnull().all()]
    if not valid_feature_names:
        print("Warning: No valid feature columns remained after processing and NaN drop.")

    return df_with_features, valid_feature_names


def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8)) # Increased size slightly
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=14)
    plt.ylabel('True label', fontsize=12); plt.xlabel('Predicted label', fontsize=12)
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0) # Improve label readability
    plt.tight_layout() # Adjust layout
    if plt.get_backend().lower() != 'agg' and (not plt.isinteractive() or 'inline' not in plt.get_backend().lower()):
        plt.show(block=False); plt.pause(0.1)
    else:
        plt.show()

def plot_equity_curves(model_equity_curve: pd.Series, 
                         buy_and_hold_equity_curve: pd.Series, 
                         model_name: str, 
                         instrument: str,
                         initial_capital: float):
    """
    Plots the equity curve of the model strategy against a Buy and Hold baseline.

    Parameters:
    -----------
    model_equity_curve : pd.Series
        Pandas Series representing the portfolio value over time for the ML model strategy.
        Index should be DatetimeIndex.
    buy_and_hold_equity_curve : pd.Series
        Pandas Series representing the portfolio value over time for the Buy & Hold strategy.
        Index should be DatetimeIndex.
    model_name : str
        Name of the ML model strategy for the plot label.
    instrument : str
        Name of the trading instrument for the plot title.
    initial_capital : float
        The starting capital for the backtest, used in the y-axis label.
    """
    plt.figure(figsize=(14, 7))
    
    if not model_equity_curve.empty:
        model_equity_curve.plot(label=f'{model_name} Strategy Equity', lw=1.5)
    else:
        print(f"Warning: '{model_name} Strategy Equity' curve is empty, not plotting.")

    if not buy_and_hold_equity_curve.empty:
        buy_and_hold_equity_curve.plot(label='Buy and Hold Equity', linestyle='--', color='grey', lw=1.5)
    else:
        print("Warning: 'Buy and Hold Equity' curve is empty, not plotting.")
    
    plt.title(f'Equity Curve Comparison: {model_name} vs. Buy & Hold for {instrument}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(f'Portfolio Value (Started with {initial_capital:,.0f})', fontsize=12)
    
    if not model_equity_curve.empty or not buy_and_hold_equity_curve.empty:
        plt.legend(fontsize=10)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    # Attempt to show plot in various environments
    if plt.get_backend().lower() != 'agg' and (not plt.isinteractive() or 'inline' not in plt.get_backend().lower()):
        plt.show(block=False) # Try non-blocking show for scripts
        plt.pause(0.1)        # Allow a moment for the plot to render
    else:
        plt.show() # Standard show for interactive environments (like Jupyter)

# --- Main Training Logic ---
if __name__ == "__main__":
    print("--- Starting Model Training, Tuning (F1) & Selection (P&L/Sharpe) with Smarter Logic ---")

    INITIAL_CAPITAL_FOR_BACKTEST = 100000.0 

    # --- Define the list of indicators to apply ---
    indicators_to_apply_list = [
        LaggedReturnsFeature(num_lags=LAGS, price_col='price'),
        MovingAverageFeature(window=10, ma_type='ema', price_col='price'),
        MovingAverageFeature(window=20, ma_type='ema', price_col='price'),
        MovingAverageFeature(window=50, ma_type='sma', price_col='price'),
        MACrossover(short_window=10, long_window=20, ma_type='ema', price_col='price'),
        RSIFeature(window=14, price_col='price'),
        MACDFeature(fast_period=12, slow_period=26, signal_period=9, price_col='price'),
        ATRFeature(window=14, high_col='high', low_col='low', close_col='price'),
        BollingerBandsFeatures(window=20, num_std_dev=2, price_col='price'),
        VolatilitySTDFeature(window=20, price_col='price'),
        HourFeature(cyclical_transform=True),
        DayOfWeekFeature(cyclical_transform=True)
    ]

    # 1. Fetch and Prepare Data
    data_full_with_features, feature_names = fetch_and_prepare_data(
        CONF_FILE, INSTRUMENT, DAYS_OF_DATA_TO_FETCH, END_OFFSET_DAYS, 
        GRANULARITY_FETCH, BAR_LENGTH_MODEL,
        indicators_to_apply=indicators_to_apply_list
    )
    if data_full_with_features is None or not feature_names: 
        print("Failed to fetch or prepare data/features. Exiting.")
        exit()
    print(f"\nPrepared data for {INSTRUMENT} with {len(data_full_with_features)} bars.")
    print(f"Using features: {feature_names}")

    # 2. Separate Features (X) and Target (y)
    X_orig = data_full_with_features[feature_names].copy()
    y = data_full_with_features['direction'].astype(int)
    
    if X_orig.empty or y.empty: 
        print("X or y is empty after feature prep. Exiting.")
        exit()
    unique_labels = sorted(list(y.unique()))
    if len(unique_labels) < 2: 
        print(f"Only one class ({unique_labels}) in target. Cannot train.")
        exit()
    print(f"Unique labels in y: {unique_labels}")

    # 3. Split Data
    train_gs_ratio = 0.7 
    validation_ratio = 0.15 
    test_ratio = 0.15     

    if not (abs(train_gs_ratio + validation_ratio + test_ratio - 1.0) < 1e-9 ):
        print("Error: Train_GS, Validation, and Test ratios must sum to 1.0")
        exit()

    n_samples = len(X_orig)
    train_gs_end_idx = int(n_samples * train_gs_ratio)
    val_end_idx = int(n_samples * (train_gs_ratio + validation_ratio))

    X_train_gs_orig, y_train_gs = X_orig.iloc[:train_gs_end_idx], y.iloc[:train_gs_end_idx]
    X_val_orig, y_val = X_orig.iloc[train_gs_end_idx:val_end_idx], y.iloc[train_gs_end_idx:val_end_idx]
    X_test_orig, y_test = X_orig.iloc[val_end_idx:], y.iloc[val_end_idx:]
    
    print(f"\nData split (before scaling):")
    print(f"X_train_gs_orig: {X_train_gs_orig.shape}")
    print(f"X_val_orig: {X_val_orig.shape}")
    print(f"X_test_orig: {X_test_orig.shape}")

    if X_train_gs_orig.empty or X_val_orig.empty or X_test_orig.empty: 
        print("One or more data splits are empty after initial split."); exit()

    # --- Feature Scaling ---
    print("\nScaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_gs_scaled_np = scaler.fit_transform(X_train_gs_orig)
    X_val_scaled_np = scaler.transform(X_val_orig)
    X_test_scaled_np = scaler.transform(X_test_orig)

    X_train_gs_scaled = pd.DataFrame(X_train_gs_scaled_np, index=X_train_gs_orig.index, columns=X_train_gs_orig.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled_np, index=X_val_orig.index, columns=X_val_orig.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled_np, index=X_test_orig.index, columns=X_test_orig.columns)
    print("Feature scaling complete.")
    print(f"X_train_gs_scaled (for GridSearchCV): {X_train_gs_scaled.shape}")

    # 4. Define Models and Parameter Grids for GridSearchCV
    models_param_grids = {
        "LogisticRegression": {
            'estimator': LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000, random_state=42),
            'params': {'C': [0.001, 0.01, 0.1, 1]}
        },
        "RandomForest": {
            'estimator': RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42),
            'params': {'n_estimators': [50, 100], 'max_depth': [10, 20], 'min_samples_leaf': [5, 10]}
        },
        "LightGBM": {
            'estimator': lgb.LGBMClassifier(objective='binary', class_weight='balanced', verbose=-1, n_jobs=-1, random_state=42),
            'params': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'num_leaves': [20, 31]}
        }
    }
    
    print("\n--- Stage 1: Hyperparameter Tuning Models with GridSearchCV (using F1-score) ---")
    tuned_models_after_gs = {}
    
    for model_name, config in models_param_grids.items():
        print(f"\nTuning {model_name} with F1-score...")
        tscv = TimeSeriesSplit(n_splits=CV_SPLITS_FOR_TUNING)
        grid_search = GridSearchCV(estimator=config['estimator'],
                                   param_grid=config['params'],
                                   cv=tscv, scoring='f1', n_jobs=-1, verbose=0)
        try:
            grid_search.fit(X_train_gs_scaled, y_train_gs) 
            tuned_models_after_gs[model_name] = grid_search.best_estimator_
            print(f"  Best F1 for {model_name} from GS: {grid_search.best_score_:.4f} with params: {grid_search.best_params_}")
        except Exception as e:
            print(f"  Error tuning {model_name}: {e}")
            import traceback; traceback.print_exc()

    if not tuned_models_after_gs: 
        print("No models were successfully tuned. Exiting."); exit()

    # --- Stage 2: Evaluating Tuned Models on Validation Set for P&L and Sharpe Ratio ---
    print("\n--- Stage 2: Evaluating Tuned Models on Validation Set for P&L and Sharpe Ratio ---")
    validation_performance = {} 
    
    # Data for validation calculator: original price + SCALED features
    validation_data_for_pnl_calc_orig_price = data_full_with_features.loc[X_val_orig.index][['price']].copy()
    temp_val_df_for_calc = pd.concat([validation_data_for_pnl_calc_orig_price, X_val_scaled], axis=1) # X_val_scaled is already a DataFrame
    temp_val_df_for_calc.dropna(inplace=True)

    for model_name, tuned_model_instance in tuned_models_after_gs.items():
        print(f"\nCalculating performance for tuned {model_name} on Validation Set...")
        
        performance_calculator = VectorizedPerformanceCalculator(
            symbol=INSTRUMENT,
            ml_model=tuned_model_instance, 
            initial_capital=INITIAL_CAPITAL_FOR_BACKTEST, 
            spread_cost_pct=SPREAD_COST_PCT,
            transaction_fee_pct=TRANSACTION_FEE_PCT,
            confidence_threshold=CONFIDENCE_THRESHOLD_BACKTEST, # Global config
            use_trend_filter=USE_TREND_FILTER_BACKTEST,         # Global config
            trend_filter_ma_period=TREND_FILTER_MA_PERIOD_BACKTEST, # Global config
            price_col_for_trend_filter='price'
        )
        
        perf_metrics = performance_calculator.calculate_performance_metrics(
            price_and_features_df=temp_val_df_for_calc, 
            feature_columns=feature_names, 
            price_column_name_for_pnl='price'
        )
        validation_performance[model_name] = perf_metrics
        print(f"  Tuned {model_name} - Validation P&L: {perf_metrics['pnl']:.2f}, Sharpe: {perf_metrics['sharpe']:.4f}")
        # print(f"    Approx Validation Trades: {(perf_metrics.get('final_signals', pd.Series(dtype=int)).diff().fillna(0) != 0).sum()}")


    valid_models_for_selection = { name: metrics for name, metrics in validation_performance.items() if np.isfinite(metrics['pnl']) }
    if not valid_models_for_selection: 
        print("No models yielded valid P&L on validation. Exiting."); exit()
        
    best_model_name_after_val = max(valid_models_for_selection, key=lambda name: valid_models_for_selection[name]['pnl'])
    overall_best_model = tuned_models_after_gs[best_model_name_after_val] 
    best_val_pnl = valid_models_for_selection[best_model_name_after_val]['pnl']
    best_val_sharpe_for_display = valid_models_for_selection[best_model_name_after_val]['sharpe']
        
    print(f"\n--- Overall Best Model Selected (based on Validation Set P&L): {best_model_name_after_val} ---")
    print(f"Best Validation Set P&L: {best_val_pnl:.2f} (Sharpe: {best_val_sharpe_for_display:.4f})")

    # --- Stage 3: Final Evaluation of Overall Best Model on Test Set ---
    print(f"\n--- Stage 3: Final Evaluation of {best_model_name_after_val} on Test Set ---")
    
    test_data_for_pnl_calc_orig_price = data_full_with_features.loc[X_test_orig.index][['price']].copy()
    temp_test_df_for_calc = pd.concat([test_data_for_pnl_calc_orig_price, X_test_scaled], axis=1) # X_test_scaled is DataFrame
    temp_test_df_for_calc.dropna(inplace=True)

    final_performance_calculator = VectorizedPerformanceCalculator(
        symbol=INSTRUMENT,
        ml_model=overall_best_model, 
        initial_capital=INITIAL_CAPITAL_FOR_BACKTEST,
        spread_cost_pct=SPREAD_COST_PCT,
        transaction_fee_pct=TRANSACTION_FEE_PCT,
        confidence_threshold=CONFIDENCE_THRESHOLD_BACKTEST,
        use_trend_filter=USE_TREND_FILTER_BACKTEST,
        trend_filter_ma_period=TREND_FILTER_MA_PERIOD_BACKTEST,
        price_col_for_trend_filter='price'
    )
    final_perf_metrics_on_test = final_performance_calculator.calculate_performance_metrics(
        price_and_features_df=temp_test_df_for_calc, 
        feature_columns=feature_names,
        price_column_name_for_pnl='price'
    )
    model_equity_curve_test = final_perf_metrics_on_test["equity_curve"]
    final_signals_on_test = final_perf_metrics_on_test.get("final_signals", pd.Series(dtype=int))
    num_trades_on_test = (final_signals_on_test.diff().fillna(0) != 0).sum()

    print(f"Final {best_model_name_after_val} - Test Set P&L: {final_perf_metrics_on_test['pnl']:.2f}, Sharpe Ratio: {final_perf_metrics_on_test['sharpe']:.4f}")
    print(f"  Approximate number of trades on test set: {num_trades_on_test}")

    test_set_prices_for_bh = data_full_with_features.loc[X_test_orig.index, 'price'].copy() 
    if test_set_prices_for_bh.empty or len(test_set_prices_for_bh) < 2:
        print("Warning: Not enough price data in test set for Buy & Hold equity curve.")
        buy_and_hold_equity_test = pd.Series(dtype=float) 
    else:
        buy_and_hold_returns_test = np.log(test_set_prices_for_bh / test_set_prices_for_bh.shift(1)).fillna(0)
        buy_and_hold_cumulative_returns_test = buy_and_hold_returns_test.cumsum()
        buy_and_hold_equity_test = pd.Series(
            INITIAL_CAPITAL_FOR_BACKTEST * np.exp(buy_and_hold_cumulative_returns_test.values), 
            index=buy_and_hold_cumulative_returns_test.index
        )
        if not buy_and_hold_equity_test.empty:
            buy_and_hold_equity_test.iloc[0] = INITIAL_CAPITAL_FOR_BACKTEST 
        else: buy_and_hold_equity_test = pd.Series(dtype=float)

    print(f"\nFinal ML Metrics for {best_model_name_after_val} on Test Set (based on raw scaled X_test predictions):")
    y_test_pred_raw_final = overall_best_model.predict(X_test_scaled) 
    test_accuracy_final = accuracy_score(y_test, y_test_pred_raw_final) 
    test_report_final = classification_report(y_test, y_test_pred_raw_final, target_names=[str(l) for l in unique_labels], zero_division=0)
    print(f"  Accuracy: {test_accuracy_final:.4f}")
    print(f"  Classification Report:\n{test_report_final}")
    
    cm_classes = [str(l) for l in unique_labels]
    print(f"\nPlotting Confusion Matrix for {best_model_name_after_val} on the Test Set...")
    plot_confusion_matrix(y_test, y_test_pred_raw_final, classes=cm_classes, 
                          title=f'Confusion Matrix - {best_model_name_after_val} (Raw Predictions - Test Set)')

    if not model_equity_curve_test.empty and not buy_and_hold_equity_test.empty:
        print(f"\nPlotting Equity Curves for {best_model_name_after_val} vs Buy & Hold on the Test Set...")
        plot_equity_curves(
            model_equity_curve=model_equity_curve_test,
            buy_and_hold_equity_curve=buy_and_hold_equity_test,
            model_name=best_model_name_after_val,
            instrument=INSTRUMENT,
            initial_capital=INITIAL_CAPITAL_FOR_BACKTEST
        )
    else:
        print("Could not plot equity curves due to empty data for one or both curves.")
    
    save_filename_final_model = MODEL_FILENAME_PATTERN.replace("TYPE", best_model_name_after_val.replace(" ", "_"))
    save_filename_final_scaler = save_filename_final_model.replace(".pkl", "_scaler.pkl")

    print(f"\nSaving the final best model ({best_model_name_after_val}) to {save_filename_final_model}...")
    try:
        with open(save_filename_final_model, 'wb') as f: pickle.dump(overall_best_model, f)
        print(f"Model successfully saved to {save_filename_final_model}")
        with open(save_filename_final_scaler, 'wb') as f: pickle.dump(scaler, f)
        print(f"Scaler successfully saved to {save_filename_final_scaler}")
    except Exception as e: print(f"Error saving model or scaler: {e}")

    print("\n--- Model Training, Tuning & Selection Script Finished ---")
    if plt.get_backend().lower() != 'agg' and (not plt.isinteractive() or 'inline' not in plt.get_backend().lower()):
        print("\nDisplaying plots. Close plot windows to exit script if they don't close automatically.")
        plt.show(block=True)