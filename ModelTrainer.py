# ModelTrainer.py

import pandas as pd
import numpy as np
import tpqoa # Your tpqoa.py should be accessible
from datetime import datetime, timedelta, timezone
import pickle

# Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# Sklearn utilities
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Custom modules
from VectorizedMetrics import calculate_vectorized_pnl_and_returns, calculate_sharpe_ratio, get_daily_returns_from_bar_returns # Ensure this file exists
from VectorizedPerformanceCalculator import VectorizedPerformanceCalculator # Ensure this file exists


# --- Configuration Parameters ---
CONF_FILE = "oanda.cfg"
INSTRUMENT = "EUR_USD"
MODEL_FILENAME_PATTERN = f"ml_trader_model_TYPE_VecSharpe_{INSTRUMENT.replace('_','')}_1min_5lags_trained.pkl"

# Data Fetching Parameters
DAYS_OF_DATA_TO_FETCH = 252 # e.g., ~3 months. Adjust based on data availability and strategy horizon
END_OFFSET_DAYS = 1        # Fetch data up to 1 day ago
GRANULARITY_FETCH = "S5"   # Granularity to fetch from API
BAR_LENGTH_MODEL = "1min"  # Granularity to resample to for model training

# Model & Feature Parameters
LAGS = 5
CV_SPLITS_FOR_TUNING = 3   # Number of splits for TimeSeriesSplit during GridSearchCV (e.g., 3-5)

# Cost Parameters for VectorizedSharpeCalculator (examples, adjust to be realistic)
SPREAD_COST_PCT = 0.0001 # e.g., 0.01% = 1 pip for EUR/USD if price is ~1.0000
TRANSACTION_FEE_PCT = 0.00005 # e.g., 0.005% if applicable

# --- Helper Functions ---
def fetch_and_prepare_data(conf_file, instrument, days_to_fetch, end_offset_days, granularity_fetch, bar_length_model):
    print(f"Initializing OANDA connection using {conf_file}...")
    api = tpqoa.tpqoa(conf_file)

    print(f"Fetching {days_to_fetch} days of {granularity_fetch} data for {instrument}, ending {end_offset_days} day(s) ago.")
    end_date = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=end_offset_days)
    end_date = end_date - timedelta(microseconds= end_date.microsecond)
    start_date = end_date - timedelta(days=days_to_fetch)

    print(f"Calculated fetch period: Start={start_date.isoformat()}, End={end_date.isoformat()}")
    try:
        raw_df = api.get_history(instrument=instrument, start=start_date, end=end_date, granularity=granularity_fetch, price="M", localize=False)
    except Exception as e:
        print(f"Error fetching data from OANDA: {e}"); return None
    if raw_df.empty: print("No data fetched."); return None
    
    print(f"Fetched {len(raw_df)} data points at {granularity_fetch} granularity.")
    
    # Determine price column for resampling (prefer 'c', fallback to instrument name)
    price_col_for_resample = None
    if 'c' in raw_df.columns:
        price_col_for_resample = 'c'
    elif instrument in raw_df.columns: # From ConTrader, 'c' is renamed to instrument
        price_col_for_resample = instrument
    
    if price_col_for_resample is None: 
        print(f"Price column ('c' or '{instrument}') not found in fetched data. Columns: {raw_df.columns}"); return None

    print(f"Resampling data to {bar_length_model} bars using column '{price_col_for_resample}'...")
    # Ensure the column exists before trying to resample it
    if price_col_for_resample not in raw_df.columns:
        print(f"Column '{price_col_for_resample}' does not exist in raw_df for resampling. Available: {raw_df.columns}"); return None

    resampled_data = raw_df[[price_col_for_resample]].resample(pd.to_timedelta(bar_length_model), label="right").last()
    resampled_data.rename(columns={price_col_for_resample: "price"}, inplace=True) # Standardize to 'price'
    resampled_data.dropna(inplace=True)

    if resampled_data.empty: print(f"No data after resampling to {bar_length_model}."); return None
    print(f"Resampled to {len(resampled_data)} {bar_length_model} bars.")
    
    resampled_data["returns"] = np.log(resampled_data["price"] / resampled_data["price"].shift(1))
    resampled_data.dropna(inplace=True)
    
    # Add features directly to this DataFrame
    feature_columns = []
    for lag in range(1, LAGS + 1): # LAGS should be defined globally or passed
        col = f'lag_{lag}'
        resampled_data[col] = resampled_data['returns'].shift(lag)
        feature_columns.append(col)
    
    # Define target (direction)
    resampled_data['direction'] = np.where(resampled_data['returns'].shift(-1) > 0, 1, -1)
    
    resampled_data.dropna(inplace=True) # Drop NaNs from lags and direction shift
    return resampled_data, feature_columns


def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    if plt.get_backend().lower() != 'agg' and (not plt.isinteractive() or 'inline' not in plt.get_backend().lower()):
        plt.show(block=False) # Attempt non-blocking show for scripts
        plt.pause(0.1) # Allow time for plot to render in some environments
    else:
        plt.show()


def plot_equity_curves(model_equity_curve: pd.Series, 
                       buy_and_hold_equity_curve: pd.Series, 
                       model_name: str, 
                       instrument: str,
                       initial_capital: float):
    """
    Plots the equity curve of the model strategy against a Buy and Hold baseline.
    """
    plt.figure(figsize=(14, 7))
    
    # Normalize for comparison if starting points differ slightly due to initial trade, or plot raw
    # Plotting raw equity curves directly
    model_equity_curve.plot(label=f'{model_name} Strategy Equity')
    buy_and_hold_equity_curve.plot(label='Buy and Hold Equity', linestyle='--')
    
    plt.title(f'Equity Curve: {model_name} vs. Buy and Hold for {instrument}')
    plt.xlabel('Date')
    plt.ylabel(f'Portfolio Value (Started with {initial_capital:,.0f})')
    plt.legend()
    plt.grid(True)
    if plt.get_backend().lower() != 'agg' and (not plt.isinteractive() or 'inline' not in plt.get_backend().lower()):
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.show()

# --- Main Training Logic ---
if __name__ == "__main__":
    print("--- Starting Model Training, Tuning (F1) & Selection (P&L/Sharpe) ---")

    INITIAL_CAPITAL_FOR_BACKTEST = 100000.0 # Define initial capital for backtests

    # 1. Fetch and Prepare Data ('data_full_with_features' has price, returns, features, direction)
    data_full_with_features, feature_names = fetch_and_prepare_data(
        CONF_FILE, INSTRUMENT, DAYS_OF_DATA_TO_FETCH, END_OFFSET_DAYS, 
        GRANULARITY_FETCH, BAR_LENGTH_MODEL
    )
    if data_full_with_features is None: 
        print("Failed to fetch or prepare data. Exiting.")
        exit()
    print(f"\nPrepared data for {INSTRUMENT} with {len(data_full_with_features)} bars. Features: {feature_names}")

    # 2. Separate Features (X) and Target (y)
    X = data_full_with_features[feature_names]
    y = data_full_with_features['direction'].astype(int)
    
    if X.empty or y.empty: 
        print("X or y is empty after feature prep. Exiting.")
        exit()
    unique_labels = sorted(list(y.unique()))
    if len(unique_labels) < 2: 
        print(f"Only one class ({unique_labels}) in target. Cannot train.")
        exit()
    print(f"Unique labels in y: {unique_labels}")

    # 3. Split Data (Train for GS, Validation for P&L/Sharpe comparison, Test for final)
    train_gs_ratio = 0.7 
    validation_ratio = 0.15 
    test_ratio = 0.15     

    if not (abs(train_gs_ratio + validation_ratio + test_ratio - 1.0) < 1e-9 ):
        print("Error: Train_GS, Validation, and Test ratios must sum to 1.0")
        exit()

    n_samples = len(X)
    train_gs_end_idx = int(n_samples * train_gs_ratio)
    val_end_idx = int(n_samples * (train_gs_ratio + validation_ratio))

    X_train_gs, y_train_gs = X.iloc[:train_gs_end_idx], y.iloc[:train_gs_end_idx]
    X_val, y_val = X.iloc[train_gs_end_idx:val_end_idx], y.iloc[train_gs_end_idx:val_end_idx]
    X_test, y_test = X.iloc[val_end_idx:], y.iloc[val_end_idx:]
    
    print(f"\nData split:")
    print(f"X_train_gs (for GridSearchCV): {X_train_gs.shape}")
    print(f"X_val (for P&L/Sharpe model selection): {X_val.shape}")
    print(f"X_test (final hold-out): {X_test.shape}")

    if X_train_gs.empty or X_val.empty or X_test.empty: 
        print("One or more data splits are empty.")
        exit()

    # 4. Define Models and Parameter Grids for GridSearchCV
    models_param_grids = {
        "LogisticRegression": {
            'estimator': LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000, random_state=42),
            'params': {'C': [0.01, 0.1, 1, 10]}
        },
        "RandomForest": {
            'estimator': RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42),
            'params': {'n_estimators': [50, 100], 'max_depth': [10, 20], 'min_samples_leaf': [2, 5]}
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
            grid_search.fit(X_train_gs, y_train_gs)
            tuned_models_after_gs[model_name] = grid_search.best_estimator_
            print(f"  Best F1 for {model_name} from GS: {grid_search.best_score_:.4f} with params: {grid_search.best_params_}")
        except Exception as e:
            print(f"  Error tuning {model_name}: {e}")
            import traceback
            traceback.print_exc()

    if not tuned_models_after_gs: 
        print("No models were successfully tuned. Exiting.")
        exit()

    # --- Stage 2: Evaluating Tuned Models on Validation Set for P&L and Sharpe Ratio ---
    print("\n--- Stage 2: Evaluating Tuned Models on Validation Set for P&L and Sharpe Ratio ---")
    validation_performance = {} 
    
    validation_data_for_calc = data_full_with_features.loc[X_val.index].copy()

    for model_name, tuned_model_instance in tuned_models_after_gs.items():
        print(f"\nCalculating performance for tuned {model_name} on Validation Set...")
        
        performance_calculator = VectorizedPerformanceCalculator(
            symbol=INSTRUMENT,
            ml_model=tuned_model_instance,
            initial_capital=INITIAL_CAPITAL_FOR_BACKTEST, 
            spread_cost_pct=SPREAD_COST_PCT,
            transaction_fee_pct=TRANSACTION_FEE_PCT
        )
        
        perf_metrics = performance_calculator.calculate_performance_metrics(
            price_and_features_df=validation_data_for_calc, 
            feature_columns=feature_names,
            price_column_name='price'
        )
        validation_performance[model_name] = perf_metrics
        print(f"  Tuned {model_name} - Validation P&L: {perf_metrics['pnl']:.2f}, Sharpe: {perf_metrics['sharpe']:.4f}")

    valid_models_for_selection = {
        name: metrics for name, metrics in validation_performance.items() 
        if np.isfinite(metrics['pnl']) 
    }
    if not valid_models_for_selection:
        print("No models yielded valid P&L on the validation set. Exiting.")
        exit()
        
    best_model_name_after_val = max(valid_models_for_selection, key=lambda name: valid_models_for_selection[name]['pnl'])
    overall_best_model = tuned_models_after_gs[best_model_name_after_val]
    best_val_pnl = valid_models_for_selection[best_model_name_after_val]['pnl']
    best_val_sharpe_for_display = valid_models_for_selection[best_model_name_after_val]['sharpe']
        
    print(f"\n--- Overall Best Model Selected (based on Validation Set P&L): {best_model_name_after_val} ---")
    print(f"Best Validation Set P&L: {best_val_pnl:.2f} (Sharpe: {best_val_sharpe_for_display:.4f})")

    # --- Stage 3: Final Evaluation of Overall Best Model on Test Set ---
    print(f"\n--- Stage 3: Final Evaluation of {best_model_name_after_val} on Test Set ---")
    
    test_data_for_calc = data_full_with_features.loc[X_test.index].copy()

    final_performance_calculator = VectorizedPerformanceCalculator(
        symbol=INSTRUMENT,
        ml_model=overall_best_model,
        initial_capital=INITIAL_CAPITAL_FOR_BACKTEST,
        spread_cost_pct=SPREAD_COST_PCT,
        transaction_fee_pct=TRANSACTION_FEE_PCT
    )
    final_perf_metrics_on_test = final_performance_calculator.calculate_performance_metrics(
        price_and_features_df=test_data_for_calc,
        feature_columns=feature_names,
        price_column_name='price'
    )
    model_equity_curve_test = final_perf_metrics_on_test["equity_curve"]
    print(f"Final {best_model_name_after_val} - Test Set P&L: {final_perf_metrics_on_test['pnl']:.2f}, Sharpe Ratio: {final_perf_metrics_on_test['sharpe']:.4f}")

    # Calculate Buy and Hold Equity Curve for the Test Set
    test_set_prices = data_full_with_features.loc[X_test.index, 'price']
    # Simple B&H: invest fully at the start, see value at the end.
    # Or, if log returns were used for model equity, use log returns for B&H too.
    buy_and_hold_returns_test = np.log(test_set_prices / test_set_prices.shift(1)).fillna(0)
    buy_and_hold_cumulative_returns_test = buy_and_hold_returns_test.cumsum()
    buy_and_hold_equity_test = INITIAL_CAPITAL_FOR_BACKTEST * np.exp(buy_and_hold_cumulative_returns_test)
    buy_and_hold_equity_test.iloc[0] = INITIAL_CAPITAL_FOR_BACKTEST


    print(f"\nFinal ML Metrics for {best_model_name_after_val} on Test Set:")
    y_test_pred_final = overall_best_model.predict(X_test)
    test_accuracy_final = accuracy_score(y_test, y_test_pred_final)
    test_report_final = classification_report(y_test, y_test_pred_final, target_names=[str(l) for l in unique_labels], zero_division=0)
    print(f"  Accuracy: {test_accuracy_final:.4f}")
    print(f"  Classification Report:\n{test_report_final}")
    
    cm_classes = [str(l) for l in unique_labels]
    # Plot Confusion Matrix (already defined)
    print(f"\nPlotting Confusion Matrix for {best_model_name_after_val} on the Test Set...")
    plot_confusion_matrix(y_test, y_test_pred_final, classes=cm_classes, 
                          title=f'Confusion Matrix - {best_model_name_after_val} (Test Set)')

    # Plot Equity Curves
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
        print("Could not plot equity curves due to empty data.")


    # Save the Overall Best Model
    save_filename_final = MODEL_FILENAME_PATTERN.replace("TYPE", best_model_name_after_val.replace(" ", "_") + "_PnLSelect") # Filename reflects PnL selection
    print(f"\nSaving the final best model ({best_model_name_after_val}) to {save_filename_final}...")
    try:
        with open(save_filename_final, 'wb') as f: 
            pickle.dump(overall_best_model, f)
        print(f"Model successfully saved to {save_filename_final}")
    except Exception as e: 
        print(f"Error saving model: {e}")

    print("\n--- Model Training, Tuning & Selection Script Finished ---")
    # Keep plots open if script finishes quickly and not in interactive mode
    if plt.get_backend().lower() != 'agg' and (not plt.isinteractive() or 'inline' not in plt.get_backend().lower()):
        print("\nDisplaying plots. Close plot windows to exit script if they don't close automatically.")
        plt.show(block=True) # Block at the end to keep plots visible