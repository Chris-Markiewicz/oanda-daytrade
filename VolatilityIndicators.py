# VolatilityIndicators.py
import pandas as pd
import numpy as np
from BaseIndicator import BaseIndicator
import talib

class ATRFeature(BaseIndicator): # Renamed from ATRIndicator
    def __init__(self, window: int = 14, high_col='high', low_col='low', close_col='price'):
        super().__init__(name=f"ATR_{window}")
        self.window = window
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col # 'price' is typically the close
        self.feature_names = [f'atr_{self.window}']

    def calculate(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        required_cols = [self.high_col, self.low_col, self.close_col]
        if not all(col in df_ohlcv.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_ohlcv.columns]
            raise ValueError(f"Missing required columns for ATR: {missing}")
        df = df_ohlcv.copy()
        feature_name = self.feature_names[0]
        try:
            df[feature_name] = talib.ATR(df[self.high_col].values, df[self.low_col].values, df[self.close_col].values, timeperiod=self.window)
        except Exception as e:
            print(f"Error calculating ATR with TA-Lib: {e}")
            df[feature_name] = np.nan
        return df

    def _get_params_string(self):
        return f"window={self.window}"

class BollingerBandsFeatures(BaseIndicator):
    def __init__(self, window: int = 20, num_std_dev: float = 2, price_col: str = 'price'):
        super().__init__(name=f"BB_{window}_{num_std_dev}std")
        self.window = window
        self.num_std_dev = num_std_dev
        self.price_col = price_col
        self.feature_names = [
            f'bb_bandwidth_{window}',
            f'bb_percent_b_{window}' # %B
        ]
        # Internal names for calculation
        self._sma_name = f'_bb_sma_{window}'
        self._std_name = f'_bb_std_{window}'
        self._upper_name = f'_bb_upper_{window}'
        self._lower_name = f'_bb_lower_{window}'


    def calculate(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        if self.price_col not in df_ohlcv.columns:
            raise ValueError(f"Price column '{self.price_col}' not found.")
        df = df_ohlcv.copy()

        df[self._sma_name] = df[self.price_col].rolling(window=self.window).mean()
        df[self._std_name] = df[self.price_col].rolling(window=self.window).std()
        
        df[self._upper_name] = df[self._sma_name] + (df[self._std_name] * self.num_std_dev)
        df[self._lower_name] = df[self._sma_name] - (df[self._std_name] * self.num_std_dev)
        
        # Bandwidth
        df[self.feature_names[0]] = (df[self._upper_name] - df[self._lower_name]) / df[self._sma_name]
        
        # %B (Percent B)
        df[self.feature_names[1]] = (df[self.price_col] - df[self._lower_name]) / (df[self._upper_name] - df[self._lower_name])
        df[self.feature_names[1]].replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero if upper=lower

        # Clean up temporary columns
        df.drop(columns=[self._sma_name, self._std_name, self._upper_name, self._lower_name], inplace=True, errors='ignore')
        return df

    def _get_params_string(self):
        return f"window={self.window}, std_dev={self.num_std_dev}"

class VolatilitySTDFeature(BaseIndicator): # Renamed from VolatilitySTD
    def __init__(self, window: int = 20, returns_col: str = 'returns', price_col: str = 'price'):
        super().__init__(name=f"VolatilitySTD_{window}")
        self.window = window
        self.returns_col = returns_col
        self.price_col = price_col # To calculate returns if not present
        self.feature_names = [f'vol_std_{self.window}']

    def calculate(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        df = df_ohlcv.copy()
        if self.returns_col not in df.columns:
            if self.price_col in df.columns:
                print(f"Warning: '{self.returns_col}' not found. Calculating log returns from '{self.price_col}'.")
                df[self.returns_col] = np.log(df[self.price_col] / df[self.price_col].shift(1))
            else:
                raise ValueError(f"Returns column '{self.returns_col}' (or '{self.price_col}') not found.")
        
        df[self.feature_names[0]] = df[self.returns_col].rolling(window=self.window).std()
        return df

    def _get_params_string(self):
        return f"window={self.window}, returns_col='{self.returns_col}'"