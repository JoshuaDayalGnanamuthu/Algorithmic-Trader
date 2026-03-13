from encryption import encrypt_message, decrypt_message
import robin_stocks.robinhood as rh
from Crypto.PublicKey import RSA
from dotenv import load_dotenv
from datetime import datetime, date
import smtplib
import ssl
from email.message import EmailMessage
import logging
import os
import mysql.connector as my
import time
from statistics import mean

# FIX [TODO]: Implement a more sophisticated trading strategy that combines multiple technical indicators (e.g., MACD, Bollinger Bands) and machine learning models to improve the accuracy of buy/sell signals.
# FIX [TODO]: Add features to track the performance of the bot, such as profit/loss, win rate, and average return per trade, to evaluate and optimize the strategy over time.
# FIX [TODO]: Bot should be able to run in the background, and start automatically on system boot, to ensure it doesn't miss any trading opportunities.
# FIX [TODO]: Bot should be aware of market hours and market open days and only operate during those times, to avoid unnecessary checks and potential errors when the market is closed.
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

RSI_PERIOD       = 14      # Number of periods for RSI calculation
RSI_OVERSOLD     = 30      # RSI below this → BUY alert
RSI_OVERBOUGHT   = 70      # RSI above this → SELL alert
CHECK_INTERVAL   = 300     # Seconds between checks (300 = 5 minutes)

WATCHLIST = [
    "AAPL", "TSLA", "ASTS", "NVDA", "AMZN", "RKLB",
    "MSFT", "GOOGL", "META", "AMD", "INTC",
    "RIVN", "LCID", "NIO",
    "SMCI", "AVGO", "TSM", "MU",
    "LUNR", "SPCE", "PL", "ASTS"
    "COIN", "MARA", "RIOT", "PLTR", "SOFI"
]

def LOGCONFIG(name: str, log_file: str, console: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s  [%(levelname)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

def SQLCONNECT() -> my.MySQLConnection | None:
    try:
        conn = my.connect(host=os.getenv("DATABASE_HOST"), user=os.getenv("DATABASE_USERNAME"), 
                          password=os.getenv("DATABASE_PASSWORD"), database="ALGOTRADER")
        if conn.is_connected():
            return conn
        return None
    except Exception as e:
        ERRORLOGGER(f"Failed to connect to MySQL database: {e}")
        return None

conn = SQLCONNECT()
cursor = conn.cursor() if conn else None
infolog = LOGCONFIG("INFO", "information.log")
errorlog = LOGCONFIG("ERROR", "error.log")
tradeslog = LOGCONFIG("TRADES", "trades.log", console=False)

def ERRORLOGGER(message: str) -> None:
    QUERY = "INSERT INTO ERRORS (TYPE, MESSAGE) VALUES (%s, %s)"
    VALUES = ("ERROR", message)
    cursor.execute(QUERY, VALUES)
    conn.commit()
    errorlog.error(message)

def INFOLOGGER(message: str) -> None:
    QUERY = "INSERT INTO EVENTS (TYPE, MESSAGE) VALUES (%s, %s)"
    VALUES = ("INFO", message)
    cursor.execute(QUERY, VALUES)
    conn.commit()
    infolog.info(message)

def WARNLOGGER(message: str) -> None:
    errorlog.warning(message)

def CRITICALLOGGER(message: str) -> None:
    errorlog.critical(message)

def SALESLOGGER(ticker: str, side: str, quantity: float, price: float, total: float, message: str) -> None:
    message = encrypt_message(message, PUBLIC_KEY)
    QUERY = "INSERT INTO TRADES (TICKER, SIDE, QUANTITY, PRICE, TOTAL) VALUES (%s, %s, %s, %s, %s)"
    VALUES = (ticker, side, quantity, price, total)
    cursor.execute(QUERY, VALUES)
    conn.commit()
    tradeslog.info(f"ENCRYPTED:{message}")

def DECRYPTLOGS() -> None:
    try:
        with open("trades.log", "r") as f:
            for line in f.readlines():
                if "ENCRYPTED:" in line:
                    encrypted_part = line.split("ENCRYPTED:")[1].strip()
                    try:
                        decrypted_message = decrypt_message(encrypted_part, PRIVATE_KEY)
                        print(decrypted_message)
                    except Exception as e:
                        ERRORLOGGER(f"Could not decrypt line: {e}")
    except Exception as e:
        ERRORLOGGER(f"Failed to read trades.log: {e}")

def LOGIN(username: str = USERNAME, password: str = PASSWORD) -> None:
    try:
        rh.login(username, password)
        INFOLOGGER("Logged in successfully.")
    except Exception as e:
        ERRORLOGGER(f"Login failed: {e}")
        exit(1)

def LASTTRANSACTION(ticker: str = None) -> date | None:
    orders = rh.orders.get_all_stock_orders(info=None)
    for order in orders:
        if order['state'] != 'filled':
            continue
        instrument = rh.stocks.get_instrument_by_url(order['instrument'])
        if instrument['symbol'] == ticker:
            dt = datetime.fromisoformat(order['last_transaction_at'].replace('Z', '+00:00'))
            return dt.date()
    ERRORLOGGER(f"No filled orders found for {ticker}")
    return None

def HOLDINGS() -> dict[str, list[float, date | None]]:
    holdings = rh.account.build_holdings(with_dividends=False)
    stocks = dict()
    for ticker, data in holdings.items():
        stocks[ticker] = [data["quantity"], LASTTRANSACTION(ticker)]
    return stocks

def BUYINGPOWER() -> float:
    return float(rh.account.load_account_profile()["buying_power"])

def BUYORDER(ticker: str, quantity: float, price: float) -> dict | None:
    try:
        if (price <= 0 or quantity <= 0):
            ERRORLOGGER(f"Buy order failed: Invalid price {price} or quantity {quantity}")
            return None
        if (not rh.stocks.get_latest_price(ticker)[0]):
            ERRORLOGGER(f"Buy order failed: Invalid ticker {ticker}")
            return None
        buying_power = BUYINGPOWER()
        if (price * quantity > buying_power):
            ERRORLOGGER(f"Buy order failed: Insufficient funds (need ${price*quantity:.2f}, have ${buying_power:.2f})")
            return None
        message = f"BUY ORDER: {ticker} | Price: ${price:.2f} | Quantity: {quantity} | Total Cost: ${price*quantity:.2f}"
        MAILALERT(f"BUY ALERT: {ticker}", message)
        SALESLOGGER(ticker, "BUY", quantity, price, price*quantity, message)
        return rh.orders.order_buy_limit(symbol=ticker, quantity=quantity, limitPrice=price, timeInForce='gfd')
    except Exception as e:
        ERRORLOGGER(f"Buy order failed: {e}")
        return None

def SELLORDER(ticker: str, quantity: float, price: float) -> dict | None:
    try:
        if (price <= 0 or quantity <= 0):
            ERRORLOGGER(f"Sell order failed: Invalid price {price}")
            return None
        holdings = HOLDINGS()
        if ticker not in holdings:
            ERRORLOGGER(f"Sell order failed: No holdings for {ticker}")
            return None
        current_date = date.today()
        last_transaction_date = holdings[ticker][1]
        if (current_date - last_transaction_date).days < 2:
            ERRORLOGGER(f"Sell order failed: Must hold {ticker} for more than 2 days (held {(current_date - last_transaction_date).days}d)")
            return None
        message = f"SELL ORDER: {ticker} | Price: ${price:.2f} | Quantity: {quantity} | Total Proceeds: ${price*quantity:.2f}"
        MAILALERT(f"SELL ALERT: {ticker}", message)
        SALESLOGGER(ticker, "SELL", quantity, price, price*quantity, message)
        return rh.orders.order_sell_limit(symbol=ticker, quantity=quantity, limitPrice=price, timeInForce='gfd')
    except Exception as e:
        ERRORLOGGER(f"Sell order failed: {e}")
        return None

def MAILALERT(subject: str, body: str) -> None:
    EMAIL = os.getenv("EMAIL")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    if not EMAIL or not EMAIL_PASSWORD:
        ERRORLOGGER("Mail alert failed: EMAIL or EMAIL_PASSWORD not set in .env")
        return
    message = EmailMessage()
    message.set_content(body)
    message["Subject"] = subject
    message["From"] = EMAIL
    message["To"] = EMAIL
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context()) as server:
            server.login(EMAIL, EMAIL_PASSWORD)
            server.send_message(message)
        INFOLOGGER("Email sent successfully!")
    except smtplib.SMTPAuthenticationError:
        ERRORLOGGER("Mail alert failed: Authentication error — check your App Password")

    except smtplib.SMTPException as e:
        ERRORLOGGER(f"Mail alert failed: {e}")  

def LASTPRICE(ticker: str) -> float | None:
    try:
        price = rh.stocks.get_latest_price(ticker)[0]
        return float(price) if price else None
    except Exception as e:
        ERRORLOGGER(f"Failed to get latest price for {ticker}: {e}")
        return None

def LOGOUT() -> None:
    INFOLOGGER("Logging out...")
    rh.authentication.logout()

def SQLCLOSE() -> None:
    if cursor:
        cursor.close()
    if conn and conn.is_connected():
        conn.close()

LOGIN()
print("Current Holdings:", STOCKS := HOLDINGS())
print("Current Buying Power:", BUYINGPOWER())
LASTTRANSACTION("RKLB")
LASTTRANSACTION("ASTS")
#SELLORDER("RKLB", 1, LASTPRICE("RKLB"))
LOGOUT()
SQLCLOSE()






