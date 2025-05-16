# TimeFeatures.py
import pandas as pd
import numpy as np
from BaseIndicator import BaseIndicator

class HourFeature(BaseIndicator):
    def __init__(self, cyclical_transform=False):
        super().__init__(name="HourFeature")
        self.cyclical_transform = cyclical_transform
        if cyclical_transform:
            self.feature_names = ['hour_sin', 'hour_cos']
        else:
            self.feature_names = ['hour']

    def calculate(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df_ohlcv.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex for HourFeature.")
        df = df_ohlcv.copy()
        hour = df.index.hour
        if self.cyclical_transform:
            df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
            df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        else:
            df['hour'] = hour
        return df
    
    def _get_params_string(self):
        return f"cyclical={self.cyclical_transform}"

class DayOfWeekFeature(BaseIndicator):
    def __init__(self, cyclical_transform=False):
        super().__init__(name="DayOfWeekFeature")
        self.cyclical_transform = cyclical_transform
        if cyclical_transform:
            self.feature_names = ['dayofweek_sin', 'dayofweek_cos']
        else:
            self.feature_names = ['dayofweek'] # Monday=0, Sunday=6

    def calculate(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df_ohlcv.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex for DayOfWeekFeature.")
        df = df_ohlcv.copy()
        dayofweek = df.index.dayofweek
        if self.cyclical_transform:
            # Using 7 for days in week
            df['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / 7.0)
            df['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / 7.0)
        else:
            df['dayofweek'] = dayofweek
        return df

    def _get_params_string(self):
        return f"cyclical={self.cyclical_transform}"