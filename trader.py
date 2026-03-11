import robin_stocks.robinhood as rh
from Crypto.PublicKey import RSA
from dotenv import load_dotenv
import datetime
import time
import logging
import os
from datetime import datetime
from statistics import mean

# FIX [TODO]: Logging should be more detailed, with separate log files for alerts and errors, and include timestamps and symbols in log entries.
# FIX [TODO]: Store data when stocks were bough, they can only be sold after 1 market day, so we need to know when we bought them to be able to sell them.
# FIX [TODO]: Add a feature to automatically execute trades based on the alerts, with proper risk management and order types (e.g., limit orders).
# FIX [TODO]: Add features to track the performance of the bot, such as profit/loss, win rate, and average return per trade, to evaluate and optimize the strategy over time.
# FIX [TODO]: Add file system encryption for logs and data files, to protect sensitive information in case of unauthorized access. Try implementing RSA encryption for added security.
# FIX [TODO]: Bot should be able to run in the background, and start automatically on system boot, to ensure it doesn't miss any trading opportunities.
# FIX [TODO]: Bot should be aware of market hours and market open days and only operate during those times, to avoid unnecessary checks and potential errors when the market is closed.
# FIX [TODO]: Add a feature to send alerts to a mobile device or email, so the user can be notified of trading opportunities even when they are not actively monitoring the bot.
# FIX [TODO]: Add a feature to backtest the RSI strategy using historical data, to evaluate its performance and optimize parameters before deploying it in live trading.
# FIX [TODO]: Add a GUI interface to allow users to easily configure the bot, view logs, and monitor performance without needing to interact with the code directly.
# FIX [TODO]: Incoporate ChatGPT to analyze news and social media sentiment for the stocks in the watchlist, to enhance the decision-making process and potentially improve trading performance.
# FIX [TODO]: Add a feature to automatically update the watchlist based on certain criteria, such as stocks with high volatility or strong technical indicators, to ensure the bot is always monitoring relevant trading opportunities.
# FIX [TODO]: Test feature to analyze what would have happened if we followed through on all the decisions the bot made, to evaluate the effectiveness of the strategy before actually executing trades in a live environment.
# FIX [TODO]: Add a feature to automatically adjust the RSI thresholds based on market conditions, such as increasing the oversold threshold during a strong downtrend, to improve the adaptability of the strategy.
# FIX [TODO]: Try including Modular Neural Networks to predict the future price of the stocks in the watchlist, and use those predictions to enhance the decision-making process of the bot, potentially improving trading performance.
# FIX [TODO]: Create Algorithm that decices how mmuch cash to allocate to each trade based on the current portfolio value, risk tolerance, and the strength of the trading signal, to optimize position sizing and manage risk effectively.
# PROJECT TIMELINE: Finish Over Spring Break??

load_dotenv("credentials.env") # Load environment variables from .env file
USERNAME = os.getenv("USERNAME") # Set your Robinhood username as an environment variable
PASSWORD = os.getenv("PASSWORD") # Set your Robinhood password as an environment variable

PRIVATE_KEY = os.getenv("PRIVATE_KEY").replace("\\n", "\n") # Set your RSA private key as an environment variable
PUBLIC_KEY = os.getenv("PUBLIC_KEY").replace("\\n", "\n") # Set your RSA public key as an environment variable
PRIVATE_KEY = RSA.import_key(PRIVATE_KEY)
PUBLIC_KEY = RSA.import_key(PUBLIC_KEY)

WATCHLIST = [
    "AAPL", "TSLA", "ASTS", "NVDA", "AMZN", "RKLB",
    "MSFT", "GOOGL", "META", "AMD", "INTC",
    "RIVN", "LCID", "NIO",
    "SMCI", "AVGO", "TSM", "MU",
    "LUNR", "SPCE", "PL", "ASTS"
    "COIN", "MARA", "RIOT", "PLTR", "SOFI"
]

def LOGIN(username = USERNAME, password = PASSWORD) -> None:
    try:
        rh.login(username, password)
        print("Logged in successfully.")
    except Exception as e:
        print(f"Login failed: {e}")
        exit(1)

def HOLDINGS() -> dict[str, float]:
    holdings = rh.account.build_holdings(with_dividends=False)
    stocks = dict()
    for ticker, data in holdings.items():
        stocks[ticker] = data["quantity"]
    return stocks

def BUYINGPOWER() -> float:
    return float(rh.account.load_account_profile()["buying_power"])

LOGIN()
print("Current Holdings:", STOCKS := HOLDINGS())
print("Current Buying Power:", BUYINGPOWER())
# HOLDINGS = rh.account.build_holdings(with_dividends=False)
# print(HOLDINGS)

RSI_PERIOD       = 14      # Number of periods for RSI calculation
RSI_OVERSOLD     = 30      # RSI below this → BUY alert
RSI_OVERBOUGHT   = 70      # RSI above this → SELL alert
CHECK_INTERVAL   = 30      # Seconds between checks (300 = 5 minutes)

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s  [%(levelname)s]  %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     handlers=[
#         logging.FileHandler("trades.log"),
#         logging.StreamHandler(),
#     ],
# )
# log = logging.getLogger(__name__)



