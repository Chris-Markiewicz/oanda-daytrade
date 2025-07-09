# Fetch historical data using tpqoa.
# Preprocess data and create features (consistent with DNNTrader).
# Split data into training, validation, and test sets.
# Normalize the features.
# Train the DNN model using functions from DNNModel.py.
# Save the trained model, mu (mean), std (standard deviation) of features, and the list of feature columns.

print("--- Starting DNN Model Training Script ---")

import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta, timezone
import pickle
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report

# Import from DNNModel.py (ensure DNNModel.py is in the same directory or Python path)
from DNNModel import set_seeds, class_weights, create_model, optimizer 

# --- Configuration ---
CONF_FILE = "oanda.cfg"  # Ensure this file exists and is configured
INSTRUMENT = "EUR_USD"
BAR_LENGTH = "20min"   # Must match DNNTrader's bar_length
WINDOW = 50            # Must match DNNTrader's window
LAGS = 5               # Must match DNNTrader's lags
DATA_DAYS = 365  # How many days of historical data to fetch for training (e.g., 3 years)

# Model Hyperparameters (can be tuned)
HIDDEN_LAYERS = 2
LAYER_UNITS = 100
DROPOUT = True
DROPOUT_RATE = 0.3
REGULARIZE = False  # Set to True and specify `reg` in create_model if needed
# LEARNING_RATE = 0.0001 # Already set in DNNModel.optimizer, but can override if creating a new optimizer here

EPOCHS = 200           # Max epochs for training
BATCH_SIZE = 64
PATIENCE = 20          # For EarlyStopping (number of epochs with no improvement)

# --- Helper Functions ---
def get_historical_data(api, instrument, bar_length_str, start_date, end_date):
    """Fetches and resamples historical data."""
    print(f"Fetching S5 data for {instrument} from {start_date} to {end_date}...")
    df = api.get_history(instrument=instrument, start=start_date, end=end_date,
                         granularity="S5", price="M", localize=False).c.dropna().to_frame()
    df.rename(columns={"c": instrument}, inplace=True)
    print(f"Resampling to {bar_length_str} bars...")
    df = df.resample(pd.to_timedelta(bar_length_str), label="right").last().dropna()
    return df

def prepare_features(df, instrument_col, window_val, lags_val):
    """Creates features and lagged variables for the model."""
    df_copy = df.copy()
    
    # Feature calculation 
    df_copy["returns"] = np.log(df_copy[instrument_col] / df_copy[instrument_col].shift())
    df_copy["dir"] = np.where(df_copy["returns"] > 0, 1, 0) # This is a feature & basis for target
    df_copy["sma"] = df_copy[instrument_col].rolling(window_val).mean() - df_copy[instrument_col].rolling(150).mean()
    df_copy["boll"] = (df_copy[instrument_col] - df_copy[instrument_col].rolling(window_val).mean()) / (df_copy[instrument_col].rolling(window_val).std() + 1e-9)
    df_copy["min"] = df_copy[instrument_col].rolling(window_val).min() / df_copy[instrument_col] - 1
    df_copy["max"] = df_copy[instrument_col].rolling(window_val).max() / df_copy[instrument_col] - 1
    df_copy["mom"] = df_copy["returns"].rolling(3).mean()
    df_copy["vol"] = df_copy["returns"].rolling(window_val).std()
    
    df_copy.dropna(inplace=True) # Drop NaNs from feature calculation

    # Lagged features
    feature_names = ["dir", "sma", "boll", "min", "max", "mom", "vol"]
    lagged_cols = []
    for f_name in feature_names:
        for lag in range(1, lags_val + 1):
            col_name = f"{f_name}_lag_{lag}"
            df_copy[col_name] = df_copy[f_name].shift(lag)
            lagged_cols.append(col_name)
    
    df_copy.dropna(inplace=True) # Drop NaNs from lagging
    return df_copy, lagged_cols

# --- Main Training Script ---
if __name__ == "__main__":
    set_seeds()  # For reproducibility

    print("\n[1/7] Loading Data...")
    # 1. Load Data
    api = tpqoa.tpqoa(CONF_FILE)
    
    # Calculate date range for historical data
    # Ensure datetime objects are naive UTC for OANDA API
    end = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    start = end - timedelta(days=DATA_DAYS)

    print(f"Attempting to fetch data for {INSTRUMENT} | Bar: {BAR_LENGTH} | Window: {WINDOW} | Lags: {LAGS}")
    raw_data = get_historical_data(api, INSTRUMENT, BAR_LENGTH, start, end)
    
    if raw_data.empty:
        print("No data fetched. Ensure OANDA credentials are correct and there's data for the period.")
        exit()
    print(f"Initial raw data shape: {raw_data.shape}")

    print("\n[2/7] Preparing Features...")
    # 2. Prepare Features
    print("Preparing features...")
    featured_data, feature_columns = prepare_features(raw_data, INSTRUMENT, WINDOW, LAGS)
    
    if featured_data.empty or not feature_columns:
        print("Data is empty after feature preparation or no feature columns generated. "
              "Check data period, window/lags values. You might need more initial data.")
        exit()
    print(f"Data shape after feature engineering: {featured_data.shape}")
    # print(f"Feature columns: {feature_columns}")

    print("\n[3/7] Defining Target and Splitting Data...")
    # 3. Define Target and Split Data (Chronological for Time Series)
    # Target: predict the direction of the *next* bar.
    featured_data["target"] = featured_data["dir"].shift(-1) # 'dir' of current bar is based on current bar's return
                                                             # So, target is next bar's 'dir'
    featured_data.dropna(inplace=True) # Remove last row due to target shift

    if featured_data.empty:
        print("Data is empty after target shifting. Not enough data. Exiting.")
        exit()

    X = featured_data[feature_columns]
    y = featured_data["target"]

    # Chronological split: 70% train, 15% validation, 15% test
    n_total = len(X)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.85)

    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train:n_val], y.iloc[n_train:n_val]
    X_test, y_test = X.iloc[n_val:], y.iloc[n_val:]

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("One of the data splits is empty. Need more data or adjust split ratios. Exiting.")
        exit()

    print("\n[4/7] Normalizing Data (Standardization)...")
    # 4. Normalize Data (Standardization)
    print("Normalizing data (calculating mu and std from training set)...")
    mu = X_train.mean()
    std = X_train.std()
    
    # Avoid division by zero if a feature has zero std (constant value)
    std = std.replace(0, 1e-6) # Replace 0 std with a tiny number

    X_train_s = (X_train - mu) / std
    X_val_s = (X_val - mu) / std
    X_test_s = (X_test - mu) / std

    print("\n[5/7] Creating and Training Model...")
    # 5. Create and Train Model
    print("Creating and training DNN model...")
    
    # Class weights for imbalanced target variable 'dir' (which is now 'target' in y_train)
    train_target_df = pd.DataFrame({'dir': y_train.astype(int)}) # class_weights expects 'dir' column
    weights = class_weights(train_target_df)
    print(f"Calculated class weights: {weights}")

    model = create_model(
        hidden_layers=HIDDEN_LAYERS,
        layer_units=LAYER_UNITS,
        dropout=DROPOUT,
        rate=DROPOUT_RATE,
        regularize=REGULARIZE,
        optimizer=optimizer, # Using the optimizer from DNNModel.py
        input_dim=X_train_s.shape[1]
    )
    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('dnn_model.keras', monitor='val_loss', save_best_only=True) # .keras format preferred

    history = model.fit(
        X_train_s, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_s, y_val),
        class_weight=weights,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    # Load the best model saved by ModelCheckpoint
    print("Loading best model from checkpoint (dnn_model.keras)...")
    best_model = tf.keras.models.load_model('dnn_model.keras')

    print("\n[6/7] Evaluating Model on Test Set...")
    # 6. Evaluate Model on Test Set
    print("Evaluating model on the test set...")
    loss, accuracy = best_model.evaluate(X_test_s, y_test, verbose=0)
    print(f"Test Set - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Classification report
    y_pred_proba = best_model.predict(X_test_s)
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    print("\nClassification Report on Test Set:")
    # Use zero_division=0 to handle cases where a class might not be predicted or present in y_test_binary
    print(classification_report(y_test, y_pred_binary, target_names=['Down (0)', 'Up (1)'], zero_division=0))

    print("\n[7/7] Saving Normalization Parameters and Feature Columns...")
    # 7. Save Normalization Parameters (mu, std) and Feature Columns
    print("Saving normalization parameters (mu.pkl, std.pkl) and feature columns (cols.pkl)...")
    mu.to_pickle("mu.pkl")
    std.to_pickle("std.pkl")
    with open('cols.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)

    print("\n--- DNN Model Training Complete ---")
    print("Trained model saved as: dnn_model.keras")
    print("Normalization mu saved as: mu.pkl")
    print("Normalization std saved as: std.pkl")
    print("Feature column list saved as: cols.pkl")
    print("These files are required by DNNTrader.py.")

print("--- DNN Model Training Script Finished ---")