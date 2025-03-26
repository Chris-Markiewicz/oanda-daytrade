import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta, timezone
import time

class ConTrader(tpqoa.tpqoa):

  def __init__(self, conf_file, instrument, bar_length):
    super().__init__(conf_file)
    self.instrument = instrument
    self.bar_length = pd.to_timedelta(bar_length) # Pandas TimeDelta 
    self.tick_data = pd.DataFrame()
    self.data = None
    self.last_bar = None

  def get_most_recent(self, days = 5): 
    now = datetime.now(timezone.utc) - timedelta(days=2)
    print(now)
    now = now - timedelta(microseconds = now.microsecond)
    end = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    start = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Fetching data from {end} to {start}")
    df = self.get_history(instrument= self.instrument,
                          start = end,
                          end = start,
                          granularity = "M1",
                          price = "M",
                          localize = False
                          ).c.dropna().to_frame()
    df.rename(columns= {"c":self.instrument}, inplace=True)
    df = df.resample(self.bar_length, label="right").last().dropna().iloc[:-1]
    self.data = df.copy()
    self.last_bar = self.data.index[-1]
    
  def on_success(self, time, bid, ask):
    print(self.ticks, end = " ", flush=True)

    # collect and store tick data
    recent_tick = pd.to_datetime(time)
    df = pd.DataFrame({self.instrument:(ask + bid) / 2},
                      index = [recent_tick])
    self.tick_data = pd.concat([self.tick_data, df])
    
    if recent_tick - self.last_bar > self.bar_length:
      self.resample_and_join()

  def resample_and_join(self):
    self.data = pd.concat([self.data, self.tick_data.resample(self.bar_length, label="right").last().ffill().iloc[:-1]])
    self.tick_data = self.tick_data.iloc[-1:]
    self.last_bar = self.data.index[-1]

td = ConTrader("oanda.cfg", "EUR_USD", "1m")
td.get_most_recent()
td.stream_data(td.instrument, stop=50)

print(td.data.tail(10))