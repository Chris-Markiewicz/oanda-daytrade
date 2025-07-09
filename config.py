# config.py

# OANDA API Configuration
OANDA_CONF_FILE = "oanda.cfg"

# Trading Parameters
INSTRUMENT = "EUR_USD"
BAR_LENGTH_TRADER = "20min"
WINDOW_TRADER = 50
LAGS_TRADER = 5
UNITS_TRADER = 100000

# Model Prediction Thresholds
PROB_THRESHOLD_LOWER = 0.47
PROB_THRESHOLD_UPPER = 0.53

# Model and Normalization File Paths
MODEL_PATH = 'dnn_model.keras'
MU_PATH = "mu.pkl"
STD_PATH = "std.pkl"
COLS_PATH = "cols.pkl"
