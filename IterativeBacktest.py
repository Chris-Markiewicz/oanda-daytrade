from IterativeBase import *


class IterativeBacktest(IterativeBase):
  ''' Class for iterative backtesting of trading strategies
  '''
  
  def go_long(self, bar, units = None, amount = None):
    if self.position == -1:
      self.buy_instrument(bar, units = -self.units)
    if units:
      self.buy_instrument(bar, units = units)
    elif amount:
      if amount == "all":
        amount = self.current_balance
      self.buy_instrument(bar, amount = amount)

  def go_short(self, bar, units = None, amount = None):
    if self.position == 1:
      self.sell_instrument(bar, units = self.units)
    if units:
      self.sell_instrument(bar, units = units)
    elif amount:
      if amount == "all":
        amount = self.current_balance
      self.sell_instrument(bar, amount = amount)

  def test_sma_strategy(self, SMA_S, SMA_L):
    '''
    Backtests an SMA crossover strategy with SMA_S (short) and SMA_L (long).
    
    Parameters
    ----------
    SMA_S: int
      moving window in bars (e.g. days) for shorter SMA
    SMA_L: int
      moving window in bars (e.g. days) for longer SMA
    '''

    # nice print
    stm = "Testing SMA strategy | {} | SMA_S = {} & SMA_L = {}".format(self.symbol, SMA_S, SMA_L)
    print(75 * "-")
    print(stm)
    print(75 * "-")

    # reset
    self.position = 0 # initial neutral position
    self.trades = 0 # no trades yet
    self.current_balance = self.initial_balance # reset initial capital
    self.get_data() # reset dataset

    # prepare data
    self.data["SMA_S"] = self.data["price"].rolling(SMA_S).mean()
    self.data["SMA_L"] = self.data["price"].rolling(SMA_L).mean()
    self.data.dropna(inplace = True)

    # sma crossover strategy
    for bar in range(len(self.data) - 1): # all bars except the last bar
      if self.data["SMA_S"].iloc[bar] > self.data["SMA_L"].iloc[bar]: #signal to go long
        if self.position in [0, -1]:
          self.go_long(bar, amount = "all") # go long with full amount
          self.position = 1 #long position
      elif self.data["SMA_S"].iloc[bar] < self.data["SMA_L"].iloc[bar]: # signal to go short
        if self.position in [0, 1]:
          self.go_short(bar, amount = "all") # go short with full amount
          self.position = -1 #short position
    self.close_position(bar+1) # close postion at the last bar

  def test_contrarian_strategy(self, window = 1):
    '''
    Backtests a simple contrarian strategy.
  
    Parameters
    ----------
    window: int
      time window (number of bars) to be considered for the strategy
    '''

    # nice print
    stm = "Testing SMA strategy | {} | Window = {}".format(self.symbol, window)
    print(75 * "-")
    print(stm)
    print(75 * "-")

    # reset
    self.position = 0
    self.trades = 0
    self.current_balance = self.initial_balance
    self.get_data()
 
    #prepare data
    self.data["rolling_returns"] = self.data["returns"].rolling(window).mean()
    self.data.dropna(inplace=True)

    #simple contrarian strategy
    for bar in range(len(self.data) - 1):
      if self.data["rolling_returns"].iloc[bar] <= 0:
        if self.position in [0, -1]:
          self.go_long(bar, amount= "all")
          self.position = 1
      elif self.data["rolling_returns"].iloc[bar] > 0:
        if self.position in [0, 1]:
          self.go_short(bar, amount= "all")
          self.position = -1
    self.close_position(bar+1)

  def test_bollinger_strategy(self, SMA, dev):
    '''
    Backtests a Bolling Bands mean-reversion strategy
    
    Parameters
    ----------
    SMA: int
      moving window in bars (e.g days) for simple moving average
    dev: int
      distance for Lower/Upper Bands in Standard Deviation Units
    '''

    # nice print
    stm = "Testing Bollinger Bands strategy | {} | SMA = {} & dev = {}".format(self.symbol, SMA, dev)
    print(75 * "-")
    print(stm)
    print(75 * "-")

    # reset
    self.position = 0
    self.trades = 0
    self.current_balance = self.initial_balance
    self.get_data()

    #prepare data
    self.data["SMA"] = self.data["price"].rolling(SMA).mean()
    self.data["Lower"] = self.data["SMA"] - self.data["price"].rolling(SMA).std() * dev
    self.data["Upper"] = self.data["SMA"] + self.data["price"].rolling(SMA).std() * dev
    self.data.dropna(inplace=True)

    # Bollinger Band Strategy
    for bar in range(len(self.data) - 1):
      if self.position == 0: # when neutral
        if self.data["price"].iloc[bar] < self.data["Lower"].iloc[bar]: # long signal
          self.go_long(bar, amount= "all")
          self.position = 1
        elif self.data["price"].iloc[bar] > self.data["Upper"].iloc[bar]: # short signal
          self.go_short(bar, amount= "all")
          self.position = -1
      elif self.position == 1: # when long
        if self.data["price"].iloc[bar] > self.data["SMA"].iloc[bar]: 
          if self.data["price"].iloc[bar] > self.data["Upper"].iloc[bar]: # short signal
            self.go_short(bar, amount="all")
            self.position = -1
          else:
            self.sell_instrument(bar, units = self.units) # go neutral
            self.position = 0
      elif self.position == -1: # when short
        if self.data["price"].iloc[bar] < self.data["SMA"].iloc[bar]: 
          if self.data["price"].iloc[bar] < self.data["Lower"].iloc[bar]:
            self.go_long(bar, amount="all")
            self.position = 1
          else:
            self.buy_instrument(bar, units = -self.units) # go neutral
            self.position = 0
    self.close_position(bar+1)

