import robin_stocks.robinhood as rh
import time
import logging
import os
from datetime import datetime
from statistics import mean


USERNAME = os.environ.get("RH_USERNAME", None) # Set your Robinhood username as an environment variable
PASSWORD = os.environ.get("RH_PASSWORD", None) # Set your Robinhood password as an environment variable
WATCHLIST = ["AAPL", "TSLA", "ASTS", "NVDA", "AMZN", "RKLB"]

RSI_PERIOD       = 14      # Number of periods for RSI calculation
RSI_OVERSOLD     = 30      # RSI below this → BUY alert
RSI_OVERBOUGHT   = 70      # RSI above this → SELL alert
CHECK_INTERVAL   = 30      # Seconds between checks (300 = 5 minutes)






def main(): 

    pass



if __name__ == "__main__":
    main()