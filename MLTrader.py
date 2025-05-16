# MLTrader.py

import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta, timezone
import time
# from sklearn.linear_model import LogisticRegression # Only needed if used as a fallback template
import pickle
import os

class MLTrader(tpqoa.tpqoa):

    def __init__(self, conf_file, instrument, bar_length, lags, units, 
                 model_filename="trained_ml_model.pkl", fallback_model_template=None): # Added fallback_model_template
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame() # For collecting current bar's ticks
        self.raw_data = None # Stores all tick data received OR historical S5 data
        self.data = None # Stores resampled bars (e.g., 1-min bars)
        self.last_bar_time = None # Timestamp of the last fully formed bar in self.data

        self.model = None # Will be loaded
        self.lags = lags
        self.units = units
        self.position = 0 # Internal position tracker: 0=neutral, 1=long, -1=short
        
        self.model_filename = model_filename
        self.fallback_model_template = fallback_model_template # e.g., LogisticRegression()

        # Feature columns must match those used in training
        self.feature_columns = [f'lag_{lag}' for lag in range(1, self.lags + 1)]
        # self.target_column = 'direction' # Not needed for prediction part

        print(f"MLTrader initialized for {self.instrument} with {self.lags} lags, trading {self.units} units.")
        self.load_model()

        if self.model is None:
            print(f"CRITICAL: Model could not be loaded from {self.model_filename}.")
            if self.fallback_model_template:
                print("Initializing with untrained fallback model template. PREDICTIONS WILL LIKELY BE RANDOM/POOR.")
                self.model = self.fallback_model_template
            else:
                print("No fallback model template provided. ML trading will likely fail.")
                # Consider exiting or disabling ML predictions if model is crucial and not loaded.

    def load_model(self):
        """Loads the ML model from a file using pickle."""
        if os.path.exists(self.model_filename):
            try:
                with open(self.model_filename, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Loaded pre-trained model from {self.model_filename}")
                if not hasattr(self.model, 'predict'):
                    print(f"Warning: Loaded object from {self.model_filename} is not a valid model. Invalidating.")
                    self.model = None
            except Exception as e:
                print(f"Error loading model from {self.model_filename}: {e}. Model not loaded.")
                self.model = None
        else:
            print(f"No pre-trained model file found at {self.model_filename}.")
            self.model = None

    def save_model(self): # Kept for completeness, but primary saving is in train_model.py
        """Saves the current ML model to a file using pickle."""
        if self.model is not None:
            try:
                with open(self.model_filename, 'wb') as f:
                    pickle.dump(self.model, f)
                print(f"Model saved to {self.model_filename}")
            except Exception as e:
                print(f"Error saving model to {self.model_filename}: {e}")
        else:
            print("No model instance to save.")

    def fetch_initial_data(self, days_for_features=2, granularity="S5"):
        ''' 
        Fetches a small amount of recent historical data to bootstrap features for live trading.
        This does NOT train the model.
        'days_for_features' should be enough to calculate all lags after resampling.
        e.g., if bar_length is 1min and lags is 5, you need at least 5+ mins of data.
        Fetching a couple of days of S5 data ensures enough history for resampling and lags.
        '''
        if self.model is None and self.fallback_model_template is None:
            print("No model loaded and no fallback, cannot effectively use fetched data for ML. Skipping initial data fetch for ML.")
            return False
            
        print(f"Fetching initial data ({days_for_features} days of {granularity}) to generate latest features for {self.instrument}...")
        
        # Fetch recent data
        now_utc = datetime.now(timezone.utc)
        end_api = now_utc - timedelta(microseconds=now_utc.microsecond) # Current time, truncated
        start_api = end_api - timedelta(days=days_for_features)
        
        # Convert to string format expected by tpqoa's get_history if needed, or pass datetime objects
        # tpqoa's get_history can handle datetime objects directly.

        df_hist = self.get_history(instrument=self.instrument,
                                   start=start_api,
                                   end=end_api,
                                   granularity=granularity,
                                   price="M", # Mid prices
                                   localize=False) 
        
        if df_hist.empty:
            print(f"No historical data fetched for initial features. Columns: {df_hist.columns if not df_hist.empty else 'N/A'}")
            return False
        
        # Store this S5 data as the base for raw_data if needed, or directly resample
        # self.raw_data will accumulate live ticks later.
        # For now, let's assume df_hist has 'c' column from get_history
        price_col = 'c'
        if price_col not in df_hist.columns:
            if self.instrument in df_hist.columns: # Fallback like in ConTrader
                price_col = self.instrument
            else:
                print(f"Price column ('c' or '{self.instrument}') not found in fetched historical data.")
                return False

        # This is the initial set of resampled bars
        resampled_hist_data = df_hist[[price_col]].resample(self.bar_length, label="right").last()
        resampled_hist_data.rename(columns={price_col: self.instrument}, inplace=True)
        resampled_hist_data.dropna(inplace=True)

        if resampled_hist_data.empty:
            print(f"Not enough historical data to form bars of length {self.bar_length} after resampling.")
            return False

        self.data = resampled_hist_data # This is our initial set of bars
        self.last_bar_time = self.data.index[-1]
        
        # Initialize self.raw_data with the S5 data for continuity if on_success appends to it
        # This part depends on how you want to manage raw S5 data vs live ticks
        # For simplicity, let's assume on_success will build its own raw_data if it starts empty
        # Or prime it:
        self.raw_data = df_hist[[price_col]].rename(columns={price_col: self.instrument}) if not df_hist.empty else pd.DataFrame()


        print(f"Initial data for features fetched. Last historical bar time: {self.last_bar_time}")
        print(f"Shape of initial self.data: {self.data.shape}")
        
        # Prepare features on this historical data to ensure logic is fine (optional print)
        # X_hist, _ = self.prepare_features(self.data)
        # if not X_hist.empty:
        #     print(f"Latest historical features prepared (for bar {self.data.index[-len(X_hist)-1]} predicting for {self.data.index[-len(X_hist)]}):\n{X_hist.tail(1)}")
        
        return True

    def prepare_features(self, data_df_input):
        ''' Prepares features (lags) from the input DataFrame.
            Input DataFrame ('data_df_input') should have the instrument's price column.
        '''
        if data_df_input is None or data_df_input.empty or self.instrument not in data_df_input.columns:
            return pd.DataFrame(), pd.Series() # Return empty to avoid downstream errors

        df = data_df_input.copy() # Work on a copy
        df['returns'] = np.log(df[self.instrument] / df[self.instrument].shift(1))
        
        for lag in range(1, self.lags + 1):
            df[f'lag_{lag}'] = df['returns'].shift(lag)
        
        # For prediction, we don't need the target 'direction' column here.
        # We only need the features for the *latest available complete bar* to predict the next.
        
        df_features_only = df[self.feature_columns].copy()
        df_features_only.dropna(inplace=True) # Drop rows where any lag is NaN
            
        return df_features_only # Return only the feature DataFrame

    def on_success(self, time_stamp, bid, ask, **kwargs): # timestamp from OANDA is already datetime
        # self.ticks is managed by parent tpqoa class
        print(self.ticks, end=" ", flush=True) 

        recent_tick_time = pd.to_datetime(time_stamp) # Ensure it's pandas datetime
        mid_price = (bid + ask) / 2.0
        
        new_tick_df = pd.DataFrame({self.instrument: mid_price}, index=[recent_tick_time])
        
        # Accumulate raw S5 tick data (or whatever granularity the stream provides)
        if self.raw_data is None or self.raw_data.empty:
            self.raw_data = new_tick_df
        else:
            # Ensure consistent column name for concat if raw_data was from historical fetch
            if self.instrument not in self.raw_data.columns and 'c' in self.raw_data.columns:
                 self.raw_data.rename(columns={'c': self.instrument}, inplace=True)
            self.raw_data = pd.concat([self.raw_data, new_tick_df])
        
        # Initialize last_bar_time from historical data if not already set by fetch_initial_data
        # or if fetch_initial_data failed or was skipped.
        if self.last_bar_time is None:
            if self.data is not None and not self.data.empty: # If self.data was populated by fetch_initial_data
                 self.last_bar_time = self.data.index[-1]
            else: # Try to resample current raw_data to see if we can form a bar
                potential_bars = self.raw_data.resample(self.bar_length, label="right").last().dropna()
                if not potential_bars.empty:
                    if self.instrument not in potential_bars.columns and 'price' in potential_bars.columns:
                         potential_bars.rename(columns={'price': self.instrument}, inplace=True)

                    self.data = potential_bars # Initialize self.data
                    self.last_bar_time = self.data.index[-1]
                    print(f"\n[on_success] Initialized last_bar_time to {self.last_bar_time} from streaming data.")
                else:
                    # Not enough data yet to form even one bar from stream, wait for more ticks
                    return 

        # Check if a new bar of self.bar_length should be formed
        if recent_tick_time - self.last_bar_time >= self.bar_length:
            self.resample_and_trade()

    def resample_and_trade(self):
        if self.raw_data is None or self.raw_data.empty:
            return

        # Resample all raw_data received so far to form bars
        # This ensures self.data contains all historical and newly formed bars
        newly_formed_bars = self.raw_data.resample(self.bar_length, label='right').last().dropna()

        if newly_formed_bars.empty or newly_formed_bars.index[-1] <= self.last_bar_time:
            # No new complete bar formed yet, or no bar after the last known one
            return

        self.data = newly_formed_bars # Update self.data with all bars up to now
        
        # Ensure the instrument column name is consistent
        if self.instrument not in self.data.columns and 'price' in self.data.columns:
            self.data.rename(columns={'price': self.instrument}, inplace=True)
        elif self.instrument not in self.data.columns and 'c' in self.data.columns:
            self.data.rename(columns={'c': self.instrument}, inplace=True)


        new_bar_time = self.data.index[-1]
        # print(f"\nNew bar formed. Previous: {self.last_bar_time}, Current: {new_bar_time}")
        self.last_bar_time = new_bar_time
        
        # Prepare features for all data up to the latest bar
        # The last row of X_to_predict will have features for the latest completed bar
        X_to_predict = self.prepare_features(self.data) 
        
        if X_to_predict.empty:
            # print(f"Not enough data in self.data (len {len(self.data)}) to create features for prediction for bar {self.last_bar_time}.")
            return

        # Get features for the latest bar for which we can make a prediction
        current_features_for_prediction = X_to_predict.iloc[-1:] # Last row of features
        
        if current_features_for_prediction.isnull().any().any():
            print(f"Warning: Features for prediction contain NaNs for bar ending {self.last_bar_time}. Bar data:\n{self.data.tail(self.lags + 2)}\nFeatures:\n{current_features_for_prediction}. Skipping prediction.")
            return

        if self.model is None:
            print("Model is not loaded. Cannot predict.")
            return

        try:
            # Predict for the NEXT bar's direction
            prediction = self.model.predict(current_features_for_prediction)[0]
            # The prediction is based on features from bar ending self.last_bar_time
            # This prediction is for the move from self.last_bar_time to self.last_bar_time + self.bar_length
            print(f"\nPrediction for {self.instrument} (bar ending {self.last_bar_time}, signal for next bar): {'LONG' if prediction == 1 else 'SHORT' if prediction == -1 else 'NEUTRAL'}")
            self.execute_trade_logic(prediction)
        except Exception as e:
            print(f"Error during prediction or trade execution: {e}")
            import traceback
            traceback.print_exc()


    def execute_trade_logic(self, prediction):
        print(f"TradeLogic | Current Pos: {self.position}, Signal: {'L' if prediction == 1 else 'S' if prediction == -1 else 'N'}, Units: {self.units}")

        if prediction == 1: # Signal to Go LONG
            if self.position == 0: # If neutral, open long
                print(f"Action: Go LONG {self.units} {self.instrument}")
                self.create_order(self.instrument, units=self.units, suppress=True, ret=True)
                self.position = 1
            elif self.position == -1: # If short, close short and open long (reverse)
                print(f"Action: Close SHORT, Go LONG {self.units} {self.instrument}")
                # Order to close short position (amount is 2 * self.units if short units are -self.units)
                # Or simply, an order of self.units closes a short of -self.units and opens a long of 0.
                # Then another self.units order.
                # Simpler: OANDA handles net positions. If short 10k, order for 20k long = net 10k long.
                self.create_order(self.instrument, units=2 * self.units, suppress=True, ret=True)
                self.position = 1
            # else: print("Already LONG or no change in signal. Holding.")
        elif prediction == -1: # Signal to Go SHORT
            if self.position == 0: # If neutral, open short
                print(f"Action: Go SHORT {-self.units} {self.instrument}")
                self.create_order(self.instrument, units=-self.units, suppress=True, ret=True)
                self.position = -1
            elif self.position == 1: # If long, close long and open short (reverse)
                print(f"Action: Close LONG, Go SHORT {-self.units} {self.instrument}")
                self.create_order(self.instrument, units=-2 * self.units, suppress=True, ret=True)
                self.position = -1
            # else: print("Already SHORT or no change in signal. Holding.")
        elif prediction == 0: # Signal to Go NEUTRAL (if your model predicts this)
            if self.position == 1: # If long, close it
                print(f"Action: Close LONG position")
                self.create_order(self.instrument, units=-self.units, suppress=True, ret=True)
                self.position = 0
            elif self.position == -1: # If short, close it
                print(f"Action: Close SHORT position")
                self.create_order(self.instrument, units=self.units, suppress=True, ret=True)
                self.position = 0
            # else: print("Already NEUTRAL. Holding.")
        
        self.report_balance()

    def report_balance(self):
        # ... (report_balance method remains the same as your previous working version) ...
        try:
            summary = self.get_account_summary()
            balance = summary.get('balance', 'N/A')
            pl = summary.get('pl', 'N/A') 
            unrealized_pl = summary.get('unrealizedPL', 'N/A')
            
            oanda_pos_qty = 0
            oanda_pos_side = "NEUTRAL"
            positions = self.get_positions() 

            for pos_data in positions:
                if pos_data['instrument'] == self.instrument:
                    long_units_str = pos_data.get('long', {}).get('units', '0') 
                    short_units_str = pos_data.get('short', {}).get('units', '0')
                    try: long_units_val = int(float(long_units_str))
                    except ValueError: long_units_val = 0
                    try: short_units_val = int(float(short_units_str))
                    except ValueError: short_units_val = 0

                    if long_units_val != 0:
                        oanda_pos_qty = long_units_val
                        oanda_pos_side = "LONG"
                    elif short_units_val != 0:
                        oanda_pos_qty = -short_units_val 
                        oanda_pos_side = "SHORT"
                    break 
            
            print(f"Balance: {balance} | P&L: {pl} | Unrealized P&L: {unrealized_pl} | "
                  f"OANDA {self.instrument} Pos: {oanda_pos_side} {oanda_pos_qty} | Internal Pos Tracker: {self.position}")
        except Exception as e:
            print(f"Could not retrieve/parse account balance: {type(e).__name__} - {e}")


# --- Example Usage ---
if __name__ == "__main__":
    conf_file = "oanda.cfg"
    instrument = "EUR_USD"
    bar_length_live = "1min"
    lags_live = 5 # MUST MATCH THE LAGS THE MODEL WAS TRAINED WITH
    units_to_trade = 10000 
    
    # IMPORTANT: Construct the model filename to match the one saved by train_model.py
    # Example: Assuming LightGBM was chosen and saved with PnLSelect suffix
    # You might need to make this more dynamic or configurable
    chosen_model_type_from_training = "LightGBM" # Or "RandomForest", "LogisticRegression"
    # Construct the filename based on how train_model.py saves it
    # e.g., MODEL_FILENAME_PATTERN = f"ml_trader_model_TYPE_VecSharpe_{INSTRUMENT.replace('_','')}_1min_5lags_trained.pkl"
    # model_save_file = f"ml_trader_model_{chosen_model_type_from_training}_VecSharpe_{instrument.replace('_','')}_{bar_length_live}_{lags_live}lags_trained.pkl"
    # OR, if train_model.py saved as "ml_trader_model_LightGBM_PnLSelect_EURUSD_1min_5lags_trained.pkl"
    model_save_file = f"ml_trader_model_{chosen_model_type_from_training}_PnLSelect_{instrument.replace('_','')}_{bar_length_live}_{lags_live}lags_trained.pkl"
    print(f"Attempting to load model: {model_save_file}")


    # Fallback model template (optional, if you want the script to run even if model loading fails)
    # from sklearn.linear_model import LogisticRegression
    # fallback_template = LogisticRegression() # Untrained!

    trader = MLTrader(conf_file=conf_file,
                      instrument=instrument,
                      bar_length=bar_length_live,
                      lags=lags_live,
                      units=units_to_trade,
                      model_filename=model_save_file,
                      fallback_model_template=None) # Or fallback_template

    # Fetch a small amount of initial data to bootstrap features for the first few predictions
    # Adjust 'days_for_features' based on bar_length and lags. 2 days of S5 should be plenty for 1min bars.
    if not trader.fetch_initial_data(days_for_features=2, granularity="S5"):
         print("Failed to initialize trader with historical data for features. Live predictions might be delayed or inaccurate initially.")
         # Decide if you want to exit or proceed cautiously
         # exit() 

    if trader.model is None:
        print("CRITICAL: No ML model loaded, and no fallback. Trading cannot proceed. Exiting.")
        exit()

    print("\n--- Starting Live Stream ---")
    try:
        trader.stream_data(trader.instrument, stop=500) # Stream for 500 ticks for testing
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred during streaming: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Finalizing ---")
        # Simplified closing logic based on internal position tracker
        if trader.position == 1: # Was long
            print(f"Final Close: Closing LONG position of {trader.units} {trader.instrument}.")
            trader.create_order(trader.instrument, units=-trader.units, suppress=True, ret=True)
        elif trader.position == -1: # Was short
            print(f"Final Close: Closing SHORT position of {trader.units} {trader.instrument}.")
            trader.create_order(trader.instrument, units=trader.units, suppress=True, ret=True)
        trader.position = 0
        trader.report_balance()
        print("Trading session ended.")