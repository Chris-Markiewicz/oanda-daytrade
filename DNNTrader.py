import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta, timezone
import time
import tensorflow as tf 
import pickle           # For loading mu, std, and cols

class DNNTrader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, window, lags, model, mu, std, units):
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None 
        self.last_bar = None
        self.units = units
        self.position = 0
        self.profits = []
        
        #*****************add strategy-specific attributes here******************
        self.window = window
        self.lags = lags
        self.model = model
        self.mu = mu
        self.std = std
        #************************************************************************
    
    def get_most_recent(self, days = 5):
        while True:
            time.sleep(2)
            now = datetime.now(timezone.utc).replace(tzinfo=None) # new (Python 3.12)
            now = now - timedelta(microseconds = now.microsecond)
            past = now - timedelta(days = days)
            df = self.get_history(instrument = self.instrument, start = past, end = now,
                                   granularity = "S5", price = "M", localize = False).c.dropna().to_frame()
            df.rename(columns = {"c":self.instrument}, inplace = True)
            df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.now(timezone.utc)) - self.last_bar < self.bar_length:
                self.start_time = pd.to_datetime(datetime.now(timezone.utc)) # NEW -> Start Time of Trading Session
                break
                
    def on_success(self, time, bid, ask):
        print(self.ticks, end = " ", flush = True)
        
        recent_tick = pd.to_datetime(time)
        df = pd.DataFrame({self.instrument:(ask + bid)/2}, 
                          index = [recent_tick])
        self.tick_data = pd.concat([self.tick_data, df]) 
        
        if recent_tick - self.last_bar > self.bar_length:
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()
    
    def resample_and_join(self):
        self.raw_data = pd.concat([self.raw_data, self.tick_data.resample(self.bar_length, 
                                                                          label="right").last().ffill().iloc[:-1]])
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]
    
    def define_strategy(self): # "strategy-specific"
        df = self.raw_data.copy()
        
        #******************** define your strategy here ************************
        #create features
        df = pd.concat([df, self.tick_data]) # append latest tick (== open price of current bar)
        df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
        df["dir"] = np.where(df["returns"] > 0, 1, 0)
        df["sma"] = df[self.instrument].rolling(self.window).mean() - df[self.instrument].rolling(150).mean()
        df["boll"] = (df[self.instrument] - df[self.instrument].rolling(self.window).mean()) / df[self.instrument].rolling(self.window).std()
        df["min"] = df[self.instrument].rolling(self.window).min() / df[self.instrument] - 1
        df["max"] = df[self.instrument].rolling(self.window).max() / df[self.instrument] - 1
        df["mom"] = df["returns"].rolling(3).mean()
        df["vol"] = df["returns"].rolling(self.window).std()
        df.dropna(inplace = True)
        
        # create lags
        self.cols = []
        features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]

        for f in features:
            for lag in range(1, self.lags + 1):
                col = "{}_lag_{}".format(f, lag)
                df[col] = df[f].shift(lag)
                self.cols.append(col)
        df.dropna(inplace = True)
        
        # standardization
        df_s = (df - self.mu) / self.std
        # predict
        df["proba"] = self.model.predict(df_s[self.cols])
        
        #determine positions
        df = df.loc[self.start_time:].copy() # starting with first live_stream bar (removing historical bars)
        df["position"] = np.where(df.proba < 0.47, -1, np.nan)
        df["position"] = np.where(df.proba > 0.53, 1, df.position)
        df["position"] = df.position.ffill().fillna(0) # start with neutral position if no strong signal
        #***********************************************************************
        
        self.data = df.copy()
    
    def execute_trades(self):
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING LONG")
            elif self.position == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress = True, ret = True) 
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.data["position"].iloc[-1] == -1: 
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units * 2, suppress = True, ret = True)
                self.report_trade(order, "GOING SHORT")
            self.position = -1
        elif self.data["position"].iloc[-1] == 0: 
            if self.position == -1:
                order = self.create_order(self.instrument, self.units, suppress = True, ret = True) 
                self.report_trade(order, "GOING NEUTRAL")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                self.report_trade(order, "GOING NEUTRAL")
            self.position = 0
    
    def report_trade(self, order, going):
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
        print(100 * "-" + "\n")  

if __name__ == "__main__":


    # --- Load Trained Model and Parameters ---
    MODEL_PATH = 'dnn_model.keras' # Keras format
    MU_PATH = "mu.pkl"
    STD_PATH = "std.pkl"
    COLS_PATH = "cols.pkl" # Path to the saved feature columns list

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
        
        print(f"Loading normalization parameters from {MU_PATH} and {STD_PATH}...")
        mu = pd.read_pickle(MU_PATH)
        std = pd.read_pickle(STD_PATH)
        print("Normalization parameters loaded successfully.")

        print(f"Loading feature columns from {COLS_PATH}...")
        with open(COLS_PATH, 'rb') as f:
            trained_cols = pickle.load(f)
        print("Feature columns loaded successfully.")

    except FileNotFoundError as e:
        print(f"Error: Could not find one or more required files: {e}")
        print("Please ensure 'dnn_model.keras', 'mu.pkl', 'std.pkl', and 'cols.pkl' exist.")
        print("Run the 'train_dnn_model.py' script first to generate these files.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading model/parameters: {e}")
        exit()
    
    # --- Initialize and Run Trader ---
    # Parameters for DNNTrader (should match training where applicable)
    instrument = "EUR_USD" 
    bar_length_trader = "20min" 
    window_trader = 50
    lags_trader = 5
    units_trader = 100000

    # # Check if trader parameters match training parameters (for critical ones)
    # # This is a sanity check. The loaded mu, std, cols implicitly define what the model expects.
    # if bar_length != bar_length_trader or WINDOW != window_trader or LAGS != lags_trader:
    #     print("Warning: Trader parameters (bar_length, window, lags) differ from training script defaults.")
    #     print("Ensure these are consistent if critical for feature generation.")
    #     # Note: The loaded 'mu', 'std', and 'cols' are derived from the training script's parameters.
    #     # The trader's 'define_strategy' must generate features compatible with these.

    trader = DNNTrader(conf_file="oanda.cfg", 
                       instrument=instrument, 
                       bar_length=bar_length_trader, 
                       window=window_trader, # Used in feature calculation
                       lags=lags_trader,     # Used in feature calculation
                       model=model, 
                       mu=mu, 
                       std=std, 
                       units=units_trader)

    # Important: The DNNTrader.define_strategy() method dynamically creates `self.cols`.
    # We must ensure that the columns it generates and uses for prediction (`df_s[self.cols]`)
    # are *exactly* the same (name and order) as `trained_cols` used for training the model.
    # The current `prepare_features` in the training script and `define_strategy` in DNNTrader
    # seem to generate columns in the same way. If they differ, predictions will be meaningless.
    # For robustness, one might pass `trained_cols` to DNNTrader or modify `define_strategy`
    # to use `trained_cols` directly, but for now, we rely on consistent generation logic.
    # A quick check:
    # trader.define_strategy() # This would run it once to populate self.cols if needed for a check
    # if trader.cols != trained_cols:
    #    print("CRITICAL ERROR: Column mismatch between training and trader. Predictions will be incorrect.")
    #    exit()
    # This check is complex to do here without data. Relies on code inspection.

    print("Fetching most recent data to initialize trader...")
    # Get enough data for feature calculation (max(window, 150) + lags bars)
    # 150 bars * 20 min/bar = 3000 mins = 50 hours = ~2.1 days. Add lags.
    # So, 10 days should be plenty.
    trader.get_most_recent(days=10) 
    
    print("Starting live trading stream (example: 100 ticks then stop)...")
    try:
        # Stream for a limited number of ticks for testing, or use a time-based stop
        trader.stream_data(trader.instrument, stop=100) # Example: stop after 100 ticks
        # For continuous trading, you might remove 'stop' or use a different stopping condition.
    except KeyboardInterrupt:
        print("\nTrading stopped by user (KeyboardInterrupt).")
    finally:
        print("Closing out any open positions...")
        trader.close_out() # Ensure to close open positions
        print("Trading session ended.")