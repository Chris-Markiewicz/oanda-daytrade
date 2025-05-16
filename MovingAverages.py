# MovingAverages.py
import pandas as pd
import numpy as np
from BaseIndicator import BaseIndicator

class MovingAverageFeature(BaseIndicator):
    def __init__(self, window: int, ma_type: str = 'sma', price_col: str = 'price'):
        super().__init__(name=f"{ma_type.upper()}_{window}")
        if window <= 0: raise ValueError("Window must be positive.")
        if ma_type.lower() not in ['sma', 'ema']: raise ValueError("ma_type must be 'sma' or 'ema'.")
            
        self.window = window
        self.ma_type = ma_type.lower()
        self.price_col = price_col
        # Feature names for the MA itself and price relative to it
        self.feature_name_ma = f'{self.ma_type}_{self.window}'
        self.feature_name_price_vs_ma = f'price_vs_{self.ma_type}_{self.window}'
        self.feature_names = [self.feature_name_ma, self.feature_name_price_vs_ma]

    def calculate(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        if self.price_col not in df_ohlcv.columns:
            raise ValueError(f"Price column '{self.price_col}' not found.")
        df = df_ohlcv.copy()
        if self.ma_type == 'sma':
            df[self.feature_name_ma] = df[self.price_col].rolling(window=self.window).mean()
        elif self.ma_type == 'ema':
            df[self.feature_name_ma] = df[self.price_col].ewm(span=self.window, adjust=False).mean()
        
        df[self.feature_name_price_vs_ma] = (df[self.price_col] / df[self.feature_name_ma]) - 1
        return df

    def _get_params_string(self):
        return f"window={self.window}, type='{self.ma_type}', price_col='{self.price_col}'"

class MACrossover(BaseIndicator):
    def __init__(self, short_window: int, long_window: int, ma_type: str = 'ema', price_col: str = 'price'):
        super().__init__(name=f"{ma_type.upper()}_{short_window}_{long_window}_Cross")
        if not (0 < short_window < long_window): raise ValueError("Windows: 0 < short < long required.")
        self.short_window = short_window
        self.long_window = long_window
        self.ma_type = ma_type.lower()
        self.price_col = price_col
        self.feature_names = [f'{self.ma_type}_{short_window}_{long_window}_cross']

    def calculate(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        if self.price_col not in df_ohlcv.columns:
            raise ValueError(f"Price column '{self.price_col}' not found.")
        df = df_ohlcv.copy()
        feature_name = self.feature_names[0]
        
        if self.ma_type == 'sma':
            short_ma = df[self.price_col].rolling(window=self.short_window).mean()
            long_ma = df[self.price_col].rolling(window=self.long_window).mean()
        elif self.ma_type == 'ema':
            short_ma = df[self.price_col].ewm(span=self.short_window, adjust=False).mean()
            long_ma = df[self.price_col].ewm(span=self.long_window, adjust=False).mean()
        else:
            raise ValueError("ma_type must be 'sma' or 'ema'.")
            
        df[feature_name] = np.where(short_ma > long_ma, 1, -1)
        return df

    def _get_params_string(self):
        return f"short={self.short_window}, long={self.long_window}, type='{self.ma_type}'"

class SlopeMA(BaseIndicator):
    def __init__(self, window: int, slope_period: int = 1, ma_type: str = 'ema', price_col: str = 'price'):
        super().__init__(name=f"{ma_type.upper()}{window}_Slope{slope_period}")
        self.window = window
        self.slope_period = slope_period
        self.ma_type = ma_type.lower()
        self.price_col = price_col
        self.feature_names = [f'{ma_type.lower()}_{window}_slope{slope_period}']
    
    def calculate(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        if self.price_col not in df_ohlcv.columns:
            raise ValueError(f"Price column '{self.price_col}' not found.")
        df = df_ohlcv.copy()
        feature_name = self.feature_names[0]
        ma_col_name = f'_temp_{self.ma_type}_{self.window}'

        if self.ma_type == 'sma':
            df[ma_col_name] = df[self.price_col].rolling(window=self.window).mean()
        elif self.ma_type == 'ema':
            df[ma_col_name] = df[self.price_col].ewm(span=self.window, adjust=False).mean()
        else:
            raise ValueError("ma_type must be 'sma' or 'ema'.")
        
        df[feature_name] = df[ma_col_name].diff(self.slope_period)
        df.drop(columns=[ma_col_name], inplace=True) # Clean up temp column
        return df

    def _get_params_string(self):
        return f"window={self.window}, slope_period={self.slope_period}, type='{self.ma_type}'"