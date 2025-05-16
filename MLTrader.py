import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta, timezone
import time
from sklearn.linear_model import LogisticRegression # Example model
import pickle # Import pickle
import os # To check if model file exists

class MLTrader(tpqoa.tpqoa):

    def __init__(self, conf_file, instrument, bar_length, model, lags, units, model_filename="trained_ml_model.pkl"): # Added model_filename
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None
        self.last_bar_time = None

        self.initial_model = model # Store the initial model instance type
        self.model = None # Will be loaded or trained
        self.lags = lags
        self.units = units
        self.position = 0
        self.model_filename = model_filename # Path to save/load the model

        self.feature_columns = [f'lag_{lag}' for lag in range(1, self.lags + 1)]
        self.target_column = 'direction'

        print(f"MLTrader initialized for {self.instrument} with {self.lags} lags, trading {self.units} units.")
        self.load_model() # Try to load an existing model

    def _ensure_model_instance(self):
        """Ensures self.model is an instance of the initial model type if not loaded."""
        if self.model is None:
            print("No pre-trained model found or loaded. Initializing a new model instance.")
            self.model = self.initial_model # Use the passed model type

    def load_model(self):
        """Loads the ML model from a file using pickle."""
        if os.path.exists(self.model_filename):
            try:
                with open(self.model_filename, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Loaded pre-trained model from {self.model_filename}")
                if not hasattr(self.model, 'predict'): # Basic check
                    print(f"Warning: Loaded object from {self.model_filename} might not be a valid model. Re-initializing.")
                    self.model = None # Invalidate if it's not a model
                    self._ensure_model_instance()
            except Exception as e:
                print(f"Error loading model from {self.model_filename}: {e}. Will train a new one.")
                self.model = None # Ensure model is None if loading fails
                self._ensure_model_instance()
        else:
            print(f"No pre-trained model file found at {self.model_filename}. A new model will be trained if data is provided.")
            self._ensure_model_instance()


    def save_model(self):
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

    def get_most_recent(self, days=5, granularity="S5", end_offset_days=0, force_retrain=False): # Added force_retrain
        ''' Fetches historical data to bootstrap the model.
        '''
        print(f"Fetching most recent data for {self.instrument}...")
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        now = now - timedelta(microseconds= now.microsecond)
        
        end_api = now 
        start_api = now - timedelta(days=days)

        end_str = end_api.strftime("%Y-%m-%dT%H:%M:%SZ")
        start_str = start_api.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        print(f"Fetching data from {start_str} to {end_str} with granularity {granularity}")

        df = self.get_history(instrument=self.instrument,
                              start=start_api,
                              end=end_api,
                              granularity=granularity,
                              price="M",
                              localize=False) 
        
        if df.empty:
            print("No historical data fetched. Exiting.")
            return False
            
        df = df.c.dropna().to_frame()
        df.rename(columns={"c": self.instrument}, inplace=True)
        
        self.raw_data = df.copy() 
        resampled_data = self.raw_data.resample(self.bar_length, label="right").last().dropna()
        
        if resampled_data.empty:
            print(f"Not enough data to form bars of length {self.bar_length} after resampling.")
            return False

        self.data = resampled_data
        self.last_bar_time = self.data.index[-1]
        print(f"Initial data fetched. Last bar time: {self.last_bar_time}")
        print(f"Initial data shape: {self.data.shape}")
        
        # Decide whether to train the model
        if self.model is None or force_retrain or not os.path.exists(self.model_filename):
            print("Training new model (or force_retrain is True, or no model loaded).")
            self._ensure_model_instance() # Make sure self.model is an instance
            self.fit_model()
        elif self.model and os.path.exists(self.model_filename):
             print("Using pre-loaded/existing model. Skipping initial training unless force_retrain=True.")
        else: # Should not happen if load_model and _ensure_model_instance work correctly
            print("Model state is unexpected. Initializing and training.")
            self._ensure_model_instance()
            self.fit_model()

        return True

    def prepare_features(self, data_df):
        ''' Prepares features (lags) and target for the ML model. '''
        if data_df is None or data_df.empty or self.instrument not in data_df.columns:
            # print("Data not available or instrument column missing for feature preparation.")
            return pd.DataFrame(), pd.Series() # Return empty to avoid downstream errors

        df = data_df.copy()
        df['returns'] = np.log(df[self.instrument] / df[self.instrument].shift(1))
        
        for lag in range(1, self.lags + 1):
            df[f'lag_{lag}'] = df['returns'].shift(lag)
        
        df[self.target_column] = np.where(df['returns'].shift(-1) > 0, 1, -1)
        
        df.dropna(inplace=True)
        
        if df.empty or not all(col in df.columns for col in self.feature_columns) or self.target_column not in df.columns:
            # print("Not enough data to create features and target after dropping NaNs.")
            return pd.DataFrame(), pd.Series() # Return empty
            
        X = df[self.feature_columns]
        y = df[self.target_column]
        return X, y

    def fit_model(self):
        ''' Trains the ML model and saves it. '''
        self._ensure_model_instance() # Ensure self.model is a valid model object

        if self.data is None or self.data.empty:
            print("No data available to train the model.")
            return

        print("Preparing features and training model...")
        X, y = self.prepare_features(self.data)
        
        if X.empty or y.empty:
            print("Feature or target data is empty. Cannot train model.")
            return
        
        if len(np.unique(y)) < 2 :
            print(f"Only one class ({np.unique(y)}) present in the target variable. Cannot train model.")
            return

        try:
            self.model.fit(X, y)
            print("Model training complete.")
            if hasattr(self.model, 'coef_'):
                print(f"Model coefficients: {self.model.coef_}")
            self.save_model() # Save the trained model
        except Exception as e:
            print(f"Error during model training: {e}")


    def on_success(self, time_stamp, bid, ask, **kwargs):
        print(self.ticks, end=" ", flush=True)

        recent_tick_time = pd.to_datetime(time_stamp)
        mid_price = (ask + bid) / 2
        new_tick_df = pd.DataFrame({self.instrument: mid_price}, index=[recent_tick_time])
        
        if self.raw_data is None:
            self.raw_data = new_tick_df
        else:
            self.raw_data = pd.concat([self.raw_data, new_tick_df])
        
        if self.last_bar_time is None:
            potential_last_bar = self.raw_data.resample(self.bar_length, label="right").last().dropna()
            if not potential_last_bar.empty:
                self.last_bar_time = potential_last_bar.index[-1]
                if self.data is None: self.data = potential_last_bar # Initialize self.data if first time
                print(f"Fallback: Initialized last_bar_time to {self.last_bar_time}")
            else:
                return

        if recent_tick_time - self.last_bar_time >= self.bar_length:
            self.resample_and_trade()

    def resample_and_trade(self):
        if self.raw_data is None or self.raw_data.empty:
            # print("No raw data to resample.") # Can be noisy
            return

        # print(f"\nResampling data at {datetime.now(timezone.utc)}...") # Can be noisy
        new_bars = self.raw_data.resample(self.bar_length, label='right').last().dropna()

        if new_bars.empty or new_bars.index[-1] <= self.last_bar_time:
            return

        self.data = new_bars # Update with all bars, not just the latest one from raw_data
        newly_formed_bar_time = self.data.index[-1]
        # print(f"New bar formed. Previous last_bar_time: {self.last_bar_time}, New last_bar_time: {newly_formed_bar_time}")
        self.last_bar_time = newly_formed_bar_time
        
        X_all, _ = self.prepare_features(self.data)
        
        if X_all.empty:
            # print("No features available after preparing for prediction.")
            return

        current_features = X_all.iloc[-1:].copy()
        
        if current_features.isnull().any().any():
            print(f"Warning: Features for prediction contain NaNs for bar {self.last_bar_time}. Skipping prediction.")
            return

        # print(f"Features for prediction (bar ending {self.last_bar_time}):\n{current_features}")
        if self.model is None:
            print("Model is not trained or loaded. Cannot predict.")
            # Optionally, try to fit the model here if enough data has accumulated
            # if len(self.data) > self.lags + 20 : # Heuristic: enough data to train
            #    self.fit_model()
            return

        try:
            prediction = self.model.predict(current_features)[0]
            print(f"Prediction for {self.instrument} @ {self.last_bar_time}: {'LONG' if prediction == 1 else 'SHORT'}")
            self.execute_trade_logic(prediction)
        except Exception as e:
            print(f"Error during prediction or trade execution: {e}")

    def execute_trade_logic(self, prediction):
        print(f"TradeLogic | Current Pos: {self.position}, Signal: {'L' if prediction == 1 else 'S'}, Units: {self.units}")

        if prediction == 1:
            if self.position == 0:
                print(f"Action: Go LONG {self.units} {self.instrument}")
                self.create_order(self.instrument, units=self.units, suppress=True, ret=True)
                self.position = 1
            elif self.position == -1:
                print(f"Action: Close SHORT, Go LONG {self.units} {self.instrument}")
                self.create_order(self.instrument, units=self.units, suppress=True, ret=True) # Close short
                self.create_order(self.instrument, units=self.units, suppress=True, ret=True) # Go long
                self.position = 1
            # else: print("Already LONG. Holding.")
        elif prediction == -1:
            if self.position == 0:
                print(f"Action: Go SHORT {-self.units} {self.instrument}")
                self.create_order(self.instrument, units=-self.units, suppress=True, ret=True)
                self.position = -1
            elif self.position == 1:
                print(f"Action: Close LONG, Go SHORT {-self.units} {self.instrument}")
                self.create_order(self.instrument, units=-self.units, suppress=True, ret=True) # Close long
                self.create_order(self.instrument, units=-self.units, suppress=True, ret=True) # Go short
                self.position = -1
            # else: print("Already SHORT. Holding.")
        self.report_balance() # Report after potential trade

    def report_balance(self):
        try:
            # OANDA specific way to get balance and P&L
            summary = self.get_account_summary()
            balance = summary.get('balance', 'N/A')
            pl = summary.get('pl', 'N/A') # Profit/Loss
            unrealized_pl = summary.get('unrealizedPL', 'N/A')
            
            # Get current position details from OANDA
            oanda_pos_qty = 0
            oanda_pos_side = "NEUTRAL"
            positions = self.get_positions() # This method is part of tpqoa or your base class

            for pos_data in positions:
                if pos_data['instrument'] == self.instrument:
                    long_units_str = pos_data.get('long', {}).get('units', '0') # More robust: handle missing 'long' key or 'units' key
                    short_units_str = pos_data.get('short', {}).get('units', '0') # More robust: handle missing 'short' key or 'units' key

                    # Convert to float first, then to int.
                    # This handles cases like '0.0', '100.0', or even just '100'.
                    try:
                        long_units_val = int(float(long_units_str))
                    except ValueError:
                        print(f"Warning: Could not parse long units string '{long_units_str}' to float/int. Defaulting to 0.")
                        long_units_val = 0
                    
                    try:
                        short_units_val = int(float(short_units_str))
                    except ValueError:
                        print(f"Warning: Could not parse short units string '{short_units_str}' to float/int. Defaulting to 0.")
                        short_units_val = 0

                    if long_units_val != 0:
                        oanda_pos_qty = long_units_val
                        oanda_pos_side = "LONG"
                    elif short_units_val != 0:
                        # OANDA short units are typically positive numbers representing the quantity short.
                        # The 'short' part of the position data indicates the direction.
                        # If your API returns them as negative, this is fine.
                        # If they are positive, and you want oanda_pos_qty to be negative for shorts:
                        oanda_pos_qty = -short_units_val # Or just short_units_val if they are already negative
                        oanda_pos_side = "SHORT"
                    break # Found instrument, no need to check other positions
            
            print(f"Balance: {balance} | P&L: {pl} | Unrealized P&L: {unrealized_pl} | "
                  f"OANDA {self.instrument} Pos: {oanda_pos_side} {oanda_pos_qty} | Internal Pos Tracker: {self.position}")

        except Exception as e:
            # Print the specific exception for better debugging
            print(f"Could not retrieve/parse account balance: {type(e).__name__} - {e}")
            # Optionally, print traceback for more detail:
            # import traceback
            # traceback.print_exc()


# --- Example Usage ---
if __name__ == "__main__":
    conf_file = "oanda.cfg"
    instrument = "EUR_USD"
    bar_length_live = "1min" # Changed to 1min for more frequent bars
    lags = 5
    units_to_trade = 10000 # Standard lot for forex example
    model_save_file = f"ml_trader_model_{instrument.replace('_','')}_{bar_length_live}_{lags}lags.pkl"

    # Initialize the type of model you want to use
    # This model instance itself won't be used if a saved one is loaded,
    # but its type is needed if a new model has to be created.
    logistic_model_template = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)

    trader = MLTrader(conf_file=conf_file,
                      instrument=instrument,
                      bar_length=bar_length_live,
                      model=logistic_model_template, # Pass the model instance/type
                      lags=lags,
                      units=units_to_trade,
                      model_filename=model_save_file)

    # Fetch initial data. Model will be trained only if not loaded or force_retrain=True
    # Use force_retrain=True if you want to retrain even if a saved model exists
    if not trader.get_most_recent(days=120, granularity="S5", end_offset_days=0, force_retrain=False):
         print("Failed to initialize trader with historical data. Exiting.")
         exit()

    print("\n--- Starting Live Stream ---")
    try:
        # Stream for a number of ticks. For continuous, remove/increase stop.
        trader.stream_data(trader.instrument, stop=100) # Stream for 500 ticks
        # For continuous streaming:
        # while True:
        #     trader.stream_data(trader.instrument, stop=1) # Process one tick
        #     # time.sleep(0.1) # Optional small delay if needed, but on_success is event-driven
        #     # trader.report_balance() # Reporting balance too frequently can be noisy / rate-limited

    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred during streaming: {e}")
    finally:
        print("\n--- Finalizing ---")
        # Optional: Save model one last time
        # trader.save_model()
        
        print("Attempting to close any open position for", trader.instrument, "based on internal tracker.")
        
        closing_action_taken = False

        if trader.position == 1: # Internally tracked as LONG
            print(f"Internal state is LONG. Creating order to SELL {trader.units} {trader.instrument} to neutralize.")
            try:
                # Create an order to sell the units we think we are holding
                # Set suppress=True to avoid the large printout
                close_order_resp = trader.create_order(trader.instrument, units=-trader.units, suppress=True, ret=True)
                if close_order_resp and 'id' in close_order_resp: # Check if response seems valid
                    print(f"Offsetting SELL order placed/filled. Transaction ID (or similar): {close_order_resp.get('id', 'N/A')}, Reason: {close_order_resp.get('reason', 'N/A')}")
                    closing_action_taken = True
                elif close_order_resp:
                    print(f"Offsetting SELL order response (unconfirmed structure): {close_order_resp}")
                    closing_action_taken = True # Assume action if any response
                else:
                    print("Offsetting SELL order did not return a confirmation.")
            except Exception as e_close:
                print(f"Error creating offsetting SELL order: {e_close}")
                import traceback
                traceback.print_exc()

        elif trader.position == -1: # Internally tracked as SHORT
            print(f"Internal state is SHORT. Creating order to BUY {trader.units} {trader.instrument} to neutralize.")
            try:
                # Create an order to buy back the units we think we are short
                # Set suppress=True to avoid the large printout
                close_order_resp = trader.create_order(trader.instrument, units=trader.units, suppress=True, ret=True)
                if close_order_resp and 'id' in close_order_resp: # Check if response seems valid
                    print(f"Offsetting BUY order placed/filled. Transaction ID (or similar): {close_order_resp.get('id', 'N/A')}, Reason: {close_order_resp.get('reason', 'N/A')}")
                    closing_action_taken = True
                elif close_order_resp:
                    print(f"Offsetting BUY order response (unconfirmed structure): {close_order_resp}")
                    closing_action_taken = True # Assume action if any response
                else:
                    print("Offsetting BUY order did not return a confirmation.")
            except Exception as e_close:
                print(f"Error creating offsetting BUY order: {e_close}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Internal state is NEUTRAL for {trader.instrument}. No cleanup order sent based on internal tracker.")
            # Optional: Sanity Check for Neutral Position (as before)
            try:
                current_oanda_positions = trader.get_positions()
                position_on_oanda_units = 0
                position_on_oanda_instrument = None

                for pos in current_oanda_positions:
                    if pos.get('instrument') == trader.instrument:
                        long_units = int(float(pos.get('long', {}).get('units', '0')))
                        short_units = int(float(pos.get('short', {}).get('units', '0')))
                        if long_units != 0:
                            position_on_oanda_units = long_units
                        elif short_units != 0:
                            position_on_oanda_units = -short_units # OANDA short units are positive
                        position_on_oanda_instrument = trader.instrument
                        break
                
                if position_on_oanda_instrument and position_on_oanda_units != 0:
                    print(f"Sanity Check: OANDA shows a position of {position_on_oanda_units} {trader.instrument} while internal tracker is neutral.")
                    # print(f"Consider manually closing this discrepancy or adding auto-logic.")
                    # Example auto-close:
                    # print(f"Attempting to auto-close discrepancy: {-position_on_oanda_units} {trader.instrument}")
                    # trader.create_order(trader.instrument, units=-position_on_oanda_units, suppress=True, ret=True)
                    # closing_action_taken = True
                elif not position_on_oanda_instrument :
                     print(f"Sanity Check: Confirmed no open position for {trader.instrument} on OANDA.")


            except Exception as e_get_pos:
                print(f"Error during final sanity check with get_positions(): {e_get_pos}")

        # Optional: Add a small delay if an action was taken to allow OANDA to update account summary
        if closing_action_taken:
            print("Waiting a moment for account state to update after closing action...")
            time.sleep(3) # Adjust delay as needed, or remove if not necessary

        trader.position = 0 # Reset internal tracker

        trader.report_balance()
        print("Trading session ended.")