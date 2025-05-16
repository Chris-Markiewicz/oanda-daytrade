# MomentumIndicators.py
import pandas as pd
import numpy as np
from BaseIndicator import BaseIndicator
import talib

class LaggedReturnsFeature(BaseIndicator): # Renamed from LaggedReturns for clarity
    def __init__(self, num_lags: int, returns_col: str = 'returns', price_col: str = 'price'):
        super().__init__(name=f"LaggedReturns_{num_lags}")
        if num_lags <= 0: raise ValueError("Number of lags must be positive.")
        self.num_lags = num_lags
        self.returns_col = returns_col
        self.price_col = price_col # To calculate returns if not present
        self.feature_names = [f'lag_{i}' for i in range(1, num_lags + 1)]

    def calculate(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        df = df_ohlcv.copy()
        if self.returns_col not in df.columns:
            if self.price_col in df.columns:
                print(f"Warning: '{self.returns_col}' not found. Calculating log returns from '{self.price_col}'.")
                df[self.returns_col] = np.log(df[self.price_col] / df[self.price_col].shift(1))
            else:
                raise ValueError(f"Returns column '{self.returns_col}' (or '{self.price_col}') not found.")
        
        for lag in range(1, self.num_lags + 1):
            df[self.feature_names[lag-1]] = df[self.returns_col].shift(lag)
        return df

    def _get_params_string(self):
        return f"num_lags={self.num_lags}, returns_col='{self.returns_col}'"

class RSIFeature(BaseIndicator): # Renamed from RSIIndicator
    def __init__(self, window: int = 14, price_col: str = 'price'):
        super().__init__(name=f"RSI_{window}")
        self.window = window
        self.price_col = price_col
        self.feature_names = [f'rsi_{self.window}']
    
    def calculate(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        if self.price_col not in df_ohlcv.columns:
            raise ValueError(f"Price column '{self.price_col}' not found.")
        df = df_ohlcv.copy()
        feature_name = self.feature_names[0]
        try:
            import talib
            df[feature_name] = talib.RSI(df[self.price_col].values, timeperiod=self.window)
        except ImportError:
            print("TA-Lib not installed. RSIFeature requires TA-Lib. Implementing manual RSI (simplified).")
            delta = df[self.price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
            rs = gain / loss
            df[feature_name] = 100 - (100 / (1 + rs))
        except Exception as e:
            print(f"Error calculating RSI: {e}. Column will be NaN.")
            df[feature_name] = np.nan
        return df

    def _get_params_string(self):
        return f"window={self.window}, price_col='{self.price_col}'"

class MACDFeature(BaseIndicator):
    def __init__(self, fast_period=12, slow_period=26, signal_period=9, price_col='price'):
        super().__init__(name=f"MACD_{fast_period}_{slow_period}_{signal_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.price_col = price_col
        self.feature_names = [
            f'macd_line_{fast_period}_{slow_period}',
            f'macd_signal_{signal_period}',
            f'macd_hist_{fast_period}_{slow_period}_{signal_period}'
        ]

    def calculate(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        if self.price_col not in df_ohlcv.columns:
            raise ValueError(f"Price column '{self.price_col}' not found.")
        df = df_ohlcv.copy()
        try:
            macd_line, macd_signal, macd_hist = talib.MACD(
                df[self.price_col].values,
                fastperiod=self.fast_period,
                slowperiod=self.slow_period,
                signalperiod=self.signal_period
            )
            df[self.feature_names[0]] = macd_line
            df[self.feature_names[1]] = macd_signal
            df[self.feature_names[2]] = macd_hist
        except Exception as e:
            print(f"Error calculating MACD with TA-Lib: {e}")
            for name in self.feature_names: df[name] = np.nan
        return df

    def _get_params_string(self):
        return f"fast={self.fast_period}, slow={self.slow_period}, signal={self.signal_period}"