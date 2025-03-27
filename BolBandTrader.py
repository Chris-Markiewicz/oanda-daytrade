from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta, timezone
import time

class BolBandTrader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, SMA, dev, units):
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None
        self.last_bar = None
        self.units = units
        self.position = 0
        self.total_units = 0
        self.profits = []

        #********************strategy specific attributes**********************#
        self.SMA = SMA
        self.dev = dev
        #**********************************************************************#

    def get_most_recent(self, days = 5):
        while True:
            time.sleep(2)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            now = now - timedelta(microseconds= now.microsecond)
            past = now - timedelta(days= days)
            df = self.get_history(instrument= self.instrument,
                                  start = past,
                                  end = now,
                                  granularity= "S5",
                                  price= "M",
                                  localize= False
                                  ).c.dropna().to_frame()
            df.rename(columns = {"c":self.instrument}, inplace= True)
            df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.now(timezone.utc)) - self.last_bar < self.bar_length:
                break
            
    def on_success(self, time, bid, ask):
        print(self.ticks, end=" ", flush=True)
        print(self.total_units)

        #collect and store tick data
        recent_tick = pd.to_datetime(time)
        df = pd.DataFrame({self.instrument:(ask+bid)/2},
                          index = [recent_tick])
        self.tick_data = pd.concat([self.tick_data, df])

        # if a time longer than the bar_length has elapsed between last full bar and the most recent tick
        if recent_tick - self.last_bar >= self.bar_length:
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()

    def resample_and_join(self):
        self.raw_data = pd.concat([self.raw_data, self.tick_data.resample(self.bar_length,
                                                                          label="right").last().ffill().iloc[:-1]])
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]

    def define_strategy(self):  # Strategy specific
        df = self.raw_data.copy()

        # ******************** Bollinger Bands Strategy ******************** #
        df["SMA"] = df[self.instrument].rolling(self.SMA).mean()
        df["Lower"] = df["SMA"] - df[self.instrument].rolling(self.SMA).std() * self.dev
        df["Upper"] = df["SMA"] + df[self.instrument].rolling(self.SMA).std() * self.dev
        df["distance"] = df[self.instrument] - df.SMA
        df["position"] = np.where(df[self.instrument] < df.Lower, 1, np.nan)
        df["position"] = np.where(df[self.instrument] > df.Upper, -1, df["position"])
        df["position"] = np.where(df.distance * df.distance.shift(1) < 0, 0, df["position"])
        df["position"] = df.position.ffill().fillna(0)
        # ****************************************************************** #

        self.data = df.copy()


    def execute_trades(self):
        new_position = self.data["position"].iloc[-1]

        if new_position == 1:  # Going LONG
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING LONG")
                self.total_units = self.units
            elif self.position == -1:  # If already SHORT, close short & go long
                order = self.create_order(self.instrument, self.total_units + self.units, suppress=True, ret=True)
                self.report_trade(order, "REVERSING TO LONG")
                self.total_units = self.units
            self.position = 1  

        elif new_position == -1:  # Going SHORT
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING SHORT")
                self.total_units = self.units
            elif self.position == 1:  # If already LONG, close long & go short
                order = self.create_order(self.instrument, -(self.total_units + self.units), suppress=True, ret=True)
                self.report_trade(order, "REVERSING TO SHORT")
                self.total_units = self.units
            self.position = -1  

        elif new_position == 0:  # Going NEUTRAL (Closing trades)
            if self.position == -1:  # If SHORT, buy back
                order = self.create_order(self.instrument, self.total_units, suppress=True, ret=True)
                self.report_trade(order, "CLOSING SHORT")
            elif self.position == 1:  # If LONG, sell back
                order = self.create_order(self.instrument, -self.total_units, suppress=True, ret=True)
                self.report_trade(order, "CLOSING LONG")
            self.position = 0  
            self.total_units = 0  # Reset units when neutral

                
    def report_trade(self, order, going):
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100 * "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
        print(100 * "-" + "\n")

trader = BolBandTrader("oanda.cfg", "EUR_USD", bar_length= "1min", SMA = 20, dev = 1, units=100000)

trader.get_most_recent()
trader.stream_data(trader.instrument, stop=100)
if trader.position != 0:
    close_order = trader.create_order(trader.instrument, units = -trader.position * trader.total_units,
                                      suppress= True, ret= True)
    trader.report_trade(close_order, "GOING NEUTRAL")
    trader.position = 0

trader.data.tail(20)[["EUR_USD", "SMA", "Lower", "Upper"]].plot(figsize=(12,8))
plt.show()