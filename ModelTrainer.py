# train_model.py

import pandas as pd
import numpy as np
import tpqoa # Your tpqoa.py should be accessible
from datetime import datetime, timedelta, timezone
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns # For a prettier confusion matrix

# --- Configuration Parameters ---
CONF_FILE = "oanda.cfg"
INSTRUMENT = "EUR_USD"
# Differentiate model name for GridSearch version
MODEL_FILENAME = f"ml_trader_model_gs_{INSTRUMENT.replace('_','')}_1min_5lags_trained.pkl"

# Data Fetching Parameters
DAYS_OF_DATA_TO_FETCH = 60 # Increased for more robust Train/Validation/Test
END_OFFSET_DAYS = 1
GRANULARITY_FETCH = "S5"
BAR_LENGTH_MODEL = "1min"

# Model & Feature Parameters
LAGS = 5

# GridSearchCV Parameters
CV_SPLITS = 5 # Number of splits for TimeSeriesSplit

# --- Helper Functions (fetch_and_prepare_data, prepare_features_and_target - remain the same as before) ---
def fetch_and_prepare_data(conf_file, instrument, days_to_fetch, end_offset_days, granularity_fetch, bar_length_model):
    """
    Fetches historical data, resamples it, and calculates returns.
    (Same as previous version)
    """
    print(f"Initializing OANDA connection using {conf_file}...")
    api = tpqoa.tpqoa(conf_file)

    print(f"Fetching {days_to_fetch} days of {granularity_fetch} data for {instrument}, ending {end_offset_days} day(s) ago.")
    
    end_date = datetime.now(timezone.utc).replace(tzinfo=None)
    end_date = end_date - timedelta(microseconds= end_date.microsecond)
    start_date = end_date - timedelta(days=days_to_fetch)

    print(f"Calculated fetch period: Start={start_date.isoformat()}, End={end_date.isoformat()}")

    try:
        raw_df = api.get_history(
            instrument=instrument,
            start=start_date,
            end=end_date,
            granularity=granularity_fetch,
            price="M",
            localize=False 
        )
    except Exception as e:
        print(f"Error fetching data from OANDA: {e}")
        return None

    if raw_df.empty:
        print("No data fetched. Check parameters or API connection.")
        return None

    print(f"Fetched {len(raw_df)} data points at {granularity_fetch} granularity.")
    
    if 'c' not in raw_df.columns:
        print("Error: 'c' (close) column not found in fetched data. Columns are:", raw_df.columns)
        if instrument in raw_df.columns:
            price_col_for_resample = instrument
        else:
            return None
    else:
        price_col_for_resample = 'c'

    print(f"Resampling data to {bar_length_model} bars using column '{price_col_for_resample}'...")
    resampled_data = raw_df[price_col_for_resample].resample(pd.to_timedelta(bar_length_model), label="right").last().to_frame()
    resampled_data.rename(columns={price_col_for_resample: instrument}, inplace=True)
    resampled_data.dropna(inplace=True)

    if resampled_data.empty:
        print(f"No data after resampling to {bar_length_model}.")
        return None

    print(f"Resampled to {len(resampled_data)} {bar_length_model} bars.")
    
    resampled_data["returns"] = np.log(resampled_data[instrument] / resampled_data[instrument].shift(1))
    resampled_data.dropna(inplace=True)
    
    return resampled_data

def prepare_features_and_target(df, instrument_col, lags):
    """
    Prepares lagged return features and the target variable (direction of next return).
    (Same as previous version)
    """
    data_ml = df.copy()
    
    feature_columns = []
    for lag in range(1, lags + 1):
        col = f'lag_{lag}'
        data_ml[col] = data_ml['returns'].shift(lag)
        feature_columns.append(col)
    
    data_ml['direction'] = np.where(data_ml['returns'].shift(-1) > 0, 1, -1)
    data_ml.dropna(inplace=True)
    
    if data_ml.empty:
        print("Not enough data to create features and target after dropping NaNs.")
        return None, None, None

    X = data_ml[feature_columns]
    y = data_ml['direction']
    
    return X, y, data_ml.index

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# --- Main Training Logic ---
if __name__ == "__main__":
    print("--- Starting Model Training Script with GridSearchCV ---")

    # 1. Fetch and Prepare Data
    data = fetch_and_prepare_data(
        conf_file=CONF_FILE,
        instrument=INSTRUMENT,
        days_to_fetch=DAYS_OF_DATA_TO_FETCH,
        end_offset_days=END_OFFSET_DAYS,
        granularity_fetch=GRANULARITY_FETCH,
        bar_length_model=BAR_LENGTH_MODEL
    )

    if data is None:
        print("Failed to fetch or prepare data. Exiting training script.")
        exit()

    print(f"\nPrepared data for {INSTRUMENT} with {len(data)} bars and 'returns' column.")

    # 2. Prepare Features and Target
    X, y, index_ml = prepare_features_and_target(data, INSTRUMENT, LAGS)

    if X is None or y is None:
        print("Failed to prepare features/target. Exiting training script.")
        exit()

    print(f"\nCreated features (X shape: {X.shape}) and target (y shape: {y.shape}).")
    
    if len(np.unique(y)) < 2:
        print(f"Only one class ({np.unique(y)}) present in the target variable. Cannot train model.")
        exit()

    # 3. Split Data (Chronologically for Time Series: Train + combined Validation/Test)
    # GridSearchCV with TimeSeriesSplit will handle the "validation" part internally.
    # We'll reserve a final test set that GridSearchCV never sees.
    
    # Total samples for training and internal validation by GridSearchCV
    train_val_ratio = 0.8  # e.g., 80% for (training + CV for hyperparameter tuning)
    test_ratio = 0.2       # e.g., 20% for final hold-out test set
    
    # Ensure there's enough data for at least CV_SPLITS in the training part
    min_data_for_cv = CV_SPLITS * 20 # Heuristic: at least 20 samples per CV split
    if len(X) * train_val_ratio < min_data_for_cv :
        print(f"Warning: Not enough data for robust TimeSeriesSplit with {CV_SPLITS} splits. Consider fetching more data or reducing CV_SPLITS.")
        # Adjust ratios or exit if too small
        if len(X) < min_data_for_cv * 2: # Arbitrary threshold
            print("Data too small for meaningful train/test split with CV. Exiting.")
            exit()

    split_index_for_test = int(len(X) * train_val_ratio)
    
    X_train_val = X.iloc[:split_index_for_test]
    y_train_val = y.iloc[:split_index_for_test]
    
    X_test = X.iloc[split_index_for_test:]
    y_test = y.iloc[split_index_for_test:]
    index_test = index_ml[split_index_for_test:]

    print(f"\nData split overview:")
    print(f"X_train_val (for GridSearchCV) shape: {X_train_val.shape}, y_train_val shape: {y_train_val.shape}")
    print(f"X_test (final hold-out) shape: {X_test.shape}, y_test shape: {y_test.shape}")

    if X_train_val.empty or X_test.empty:
        print("Training/validation or testing set is empty. Need more data or adjust split ratio.")
        exit()

    # 4. Setup GridSearchCV for Hyperparameter Tuning
    print(f"\nSetting up GridSearchCV for Logistic Regression...")
    
    # Define the parameter grid for LogisticRegression
    # 'C' is the inverse of regularization strength; smaller values specify stronger regularization.
    # 'solver' options depend on the problem size and type.
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'], # 'liblinear' is good for smaller datasets, 'lbfgs' is a common default
        'class_weight': [None, 'balanced']
    }

    # Initialize TimeSeriesSplit
    # n_splits determines how many train/test iterations CV will do.
    # The test set in each split will be chronologically after the train set.
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    # Initialize Logistic Regression model
    log_reg = LogisticRegression(max_iter=1000, random_state=42)

    # Initialize GridSearchCV
    # Scoring can be 'accuracy', 'f1', 'roc_auc', etc. 'f1' is often good for imbalanced classes.
    grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=tscv, scoring='f1', n_jobs=-1, verbose=1)
    # n_jobs=-1 uses all available CPU cores
    # verbose=1 shows some progress

    print("Starting GridSearchCV... This may take a while.")
    try:
        grid_search.fit(X_train_val, y_train_val)
    except Exception as e:
        print(f"Error during GridSearchCV fitting: {e}")
        exit()

    print("\nGridSearchCV finished.")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best F1 score on validation sets: {grid_search.best_score_:.4f}")

    # Get the best model found by GridSearchCV
    best_model = grid_search.best_estimator_
    if hasattr(best_model, 'coef_'):
            print(f"Best model coefficients: {best_model.coef_}")


    # 5. Evaluate the Best Model on the Hold-Out Test Set
    print("\nEvaluating the best model on the hold-out test set...")
    y_pred_test = best_model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test, target_names=['Down (-1)', 'Up (1)'], zero_division=0)
    
    print(f"\nHold-Out Test Set Accuracy: {test_accuracy:.4f}")
    print("Hold-Out Test Set Classification Report:")
    print(test_report)

    # 6. Plot Confusion Matrix for the Test Set
    print("\nPlotting Confusion Matrix for the Test Set...")
    plot_confusion_matrix(y_test, y_pred_test, classes=['Down (-1)', 'Up (1)'], title='Confusion Matrix - Test Set')

    # 7. Save the Best Trained Model
    print(f"\nSaving the best trained model to {MODEL_FILENAME}...")
    try:
        with open(MODEL_FILENAME, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"Model successfully saved to {MODEL_FILENAME}")
    except Exception as e:
        print(f"Error saving model: {e}")

    print("\n--- Model Training Script Finished ---")