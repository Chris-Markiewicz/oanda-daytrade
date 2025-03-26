import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta, timezone
import time

class ConTrader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, window, units):
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None
        self.last_bar = None
        self.units = units
        self.posistion = 0
        self.profits = []

        #********************strategy specific attributes**********************#
        self.window = window
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

    def define_strategy(self): #strategy specific
        df = self.raw_data.copy()

        #********* Strategy Here ************#
        df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
        df["position"] = -np.sign(df.returns.rolling(self.window).mean())
        #************************************#

        self.data = df.copy()

    def execute_trades(self):
        if self.data["position"].iloc[-1] == 1:
            if self.posistion == 0:
                order = self.create_order(self.instrument, self.units, suppress= True, ret= True)
                self.report_trade(order, "GOING LONG")
            elif self.posistion == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress= True, ret= True)
                self.report_trade(order, "GOING LONG")
            self.posistion = 1

        elif self.data["position"].iloc[-1] == -1:
            if self.posistion == 0:
                order = self.create_order(self.instrument, -self.units, suppress= True, ret= True)
                self.report_trade(order, "GOING SHORT")
            elif self.posistion == -1:
                order = self.create_order(self.instrument, -self.units * 2, suppress= True, ret= True)
                self.report_trade(order, "GOING SHORT")
            self.posistion = -1
            
        elif self.data["position"].iloc[-1] == 0:
            if self.posistion == -1:
                order = self.create_order(self.instrument, self.units, suppress= True, ret= True)
                self.report_trade(order, "GOING NEUTRAL")
            elif self.posistion == 1:
                order = self.create_order(self.instrument, -self.units, suppress= True, ret= True)
                self.report_trade(order, "GOING NEUTRAL")
            self.posistion = 0
                
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

trader = ConTrader("oanda.cfg", "EUR_USD", bar_length= "1min", window = 1, units=100000)

trader.get_most_recent()
trader.stream_data(trader.instrument, stop=200)
if trader.posistion != 0:
    close_order = trader.create_order(trader.instrument, units = -trader.posistion * trader.units,
                                      suppress= True, ret= True)
    trader.report_trade(close_order, "GOING NEUTRAL")
    trader.posistion = 0