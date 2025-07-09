import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta, timezone
import time
import tensorflow as tf 
import pickle           # For loading mu, std, and cols
import logging
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DNNTrader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, window, lags, model, mu, std, units, trained_cols, prob_threshold_lower=0.47, prob_threshold_upper=0.53):
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
        self.trained_cols = trained_cols # Store the columns the model was trained on
        self.prob_threshold_lower = prob_threshold_lower
        self.prob_threshold_upper = prob_threshold_upper
        #************************************************************************
    
    def get_most_recent(self, days = 10):
        while True:
            time.sleep(2)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
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
        logging.info(f"Tick: {self.ticks}")
        
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
        # Note: Features for the current bar are calculated using the latest tick as the 'open' price.
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
        if df.empty:
            print("Warning: DataFrame became empty after feature calculation and dropna. Skipping strategy definition.")
            self.data = pd.DataFrame() # Ensure self.data is empty to prevent errors
            return
        
        # create lags
        features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]

        for f in features:
            for lag in range(1, self.lags + 1):
                col = "{}_lag_{}".format(f, lag)
                df[col] = df[f].shift(lag)
        df.dropna(inplace = True)
        if df.empty:
            print("Warning: DataFrame became empty after lag creation and dropna. Skipping strategy definition.")
            self.data = pd.DataFrame() # Ensure self.data is empty to prevent errors
            return
        
        # standardization
        df_s = (df - self.mu) / self.std
        # predict
        df["proba"] = self.model.predict(df_s[self.trained_cols])
        
        #determine positions
        df = df.loc[self.start_time:].copy() # starting with first live_stream bar (removing historical bars)
        df["position"] = np.where(df.proba < self.prob_threshold_lower, -1, np.nan)
        df["position"] = np.where(df.proba > self.prob_threshold_upper, 1, df.position)
        df["position"] = df.position.ffill().fillna(0) # start with neutral position if no strong signal
        #***********************************************************************
        
        self.data = df.copy()
    
    def execute_trades(self):
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                try:
                    order = self.create_order(self.instrument, self.units, suppress = True, ret = True)
                    self.report_trade(order, "GOING LONG")
                except Exception as e:
                    print(f"Error creating order (GOING LONG): {e}")
            elif self.position == -1:
                try:
                    order = self.create_order(self.instrument, self.units * 2, suppress = True, ret = True) 
                    self.report_trade(order, "GOING LONG")
                except Exception as e:
                    print(f"Error creating order (GOING LONG, closing short): {e}")
            self.position = 1
        elif self.data["position"].iloc[-1] == -1: 
            if self.position == 0:
                try:
                    order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                    self.report_trade(order, "GOING SHORT")
                except Exception as e:
                    print(f"Error creating order (GOING SHORT): {e}")
            elif self.position == 1:
                try:
                    order = self.create_order(self.instrument, -self.units * 2, suppress = True, ret = True)
                    self.report_trade(order, "GOING SHORT")
                except Exception as e:
                    print(f"Error creating order (GOING SHORT, closing long): {e}")
            self.position = -1
        elif self.data["position"].iloc[-1] == 0: 
            if self.position == -1:
                try:
                    order = self.create_order(self.instrument, self.units, suppress = True, ret = True) 
                    self.report_trade(order, "GOING NEUTRAL")
                except Exception as e:
                    print(f"Error creating order (GOING NEUTRAL, closing short): {e}")
            elif self.position == 1:
                try:
                    order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
                    self.report_trade(order, "GOING NEUTRAL")
                except Exception as e:
                    print(f"Error creating order (GOING NEUTRAL, closing long): {e}")
            self.position = 0
    
    def report_trade(self, order, going):
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        logging.info("\n" + 100* "-")
        logging.info("{} | {}".format(time, going))
        logging.info("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
        logging.info(100 * "-" + "\n")  

if __name__ == "__main__":
    # --- Load Trained Model and Parameters from config file ---
    logging.info(f"Loading model from {config.MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(config.MODEL_PATH)
        logging.info("Model loaded successfully.")
        
        logging.info(f"Loading normalization parameters from {config.MU_PATH} and {config.STD_PATH}...")
        mu = pd.read_pickle(config.MU_PATH)
        std = pd.read_pickle(config.STD_PATH)
        logging.info("Normalization parameters loaded successfully.")

        logging.info(f"Loading feature columns from {config.COLS_PATH}...")
        with open(config.COLS_PATH, 'rb') as f:
            trained_cols = pickle.load(f)
        logging.info("Feature columns loaded successfully.")

    except FileNotFoundError as e:
        logging.error(f"Error: Could not find one or more required files: {e}")
        logging.error("Please ensure the paths in 'config.py' are correct and the files exist.")
        logging.error("You may need to run a training script to generate these files.")
        exit()
    except Exception as e:
        logging.error(f"An error occurred while loading model/parameters: {e}")
        exit()
    
    # --- Initialize and Run Trader using parameters from config file ---
    trader = DNNTrader(conf_file=config.OANDA_CONF_FILE, 
                       instrument=config.INSTRUMENT, 
                       bar_length=config.BAR_LENGTH_TRADER, 
                       window=config.WINDOW_TRADER,
                       lags=config.LAGS_TRADER,
                       model=model, 
                       mu=mu, 
                       std=std, 
                       units=config.UNITS_TRADER,
                       trained_cols=trained_cols,
                       prob_threshold_lower=config.PROB_THRESHOLD_LOWER,
                       prob_threshold_upper=config.PROB_THRESHOLD_UPPER)

    logging.info("Fetching most recent data to initialize trader...")
    trader.get_most_recent(days=10) 
    
    logging.info("Starting live trading stream...")
    try:
        # Stream data for the specified instrument.
        # For continuous trading, this can be run without a 'stop' condition.
        # For testing, you might add a stop condition, e.g., trader.stream_data(trader.instrument, stop=100)
        trader.stream_data(trader.instrument)
    except KeyboardInterrupt:
        logging.info("\nTrading stopped by user (KeyboardInterrupt).")
    finally:
        logging.info("Closing out any open positions...")
        trader.close_out()
        logging.info("Trading session ended.")