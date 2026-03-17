import robin_stocks.robinhood as rh
from Crypto.PublicKey import RSA
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging
import os
import mysql.connector as my
import time
from ModularNeuralNetwork import ModularNeuralNet
import numpy as np
from Historicals import CalculateRSI, Volatility, CalculateBollinger, CalculateMACD, SafeDivide, CalculateATR
import joblib
import math

load_dotenv("credentials.env")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
MODEL = ModularNeuralNet.load_model('files/trader_model.npy')
PRIVATE_KEY = RSA.import_key(os.getenv("PRIVATE_KEY").replace("\\n", "\n"))
PUBLIC_KEY  = RSA.import_key(os.getenv("PUBLIC_KEY").replace("\\n", "\n"))
CAPITAL = 100000.0
PORTFOLIO = CAPITAL

WATCHLIST = [
    "AAPL", "TSLA", "ASTS", "NVDA", "AMZN",
    "MSFT", "GOOGL", "META", "AMD", "INTC",
    "RIVN", "RKLB", "SPY", "QQQ"]

HOLD_HOURS = 5
HOLDINGS = dict.fromkeys(WATCHLIST, 0.0)
PURCHASE_PRICES = dict.fromkeys(WATCHLIST, 0.0)
BUY_TIMESTAMPS  = dict.fromkeys(WATCHLIST, None)

SCAN_INTERVAL = 1800
TAKE_PROFIT = 0.03
STOP_LOSS  = 0.03
POSITION_SIZE = 5000
MAX_POSITIONS = 6

NO_OF_TRADES = [0]
PORTFOLIO_CHANGE = [PORTFOLIO]
scaler = joblib.load(r"files/scaler.save")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler("trades.log"), logging.StreamHandler()])

def MarketHours() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    current_time = now.hour * 100 + now.minute
    return 930 <= current_time < 1600

def SecondsUntilMarketOpen() -> float:
    now  = datetime.now()
    days_ahead = 0
    while True:
        candidate = (now + timedelta(days=days_ahead)).replace(hour=9, minute=30, second=0, microsecond=0)
        if candidate > now and candidate.weekday() < 5:
            return (candidate - now).total_seconds()
        days_ahead += 1

def MarketMinutesElapsed(buy_time: datetime, now: datetime) -> float:
    MARKET_OPEN  = (9, 30)
    MARKET_CLOSE = (16, 0)
    TRADING_HOURS_PER_DAY = 6.5
    def clamp_to_market(dt: datetime) -> float:
        """Return fractional trading hours into the day for a given datetime."""
        open_dt  = dt.replace(hour=MARKET_OPEN[0],  minute=MARKET_OPEN[1],  second=0, microsecond=0)
        close_dt = dt.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0)
        if dt <= open_dt:
            return 0.0
        if dt >= close_dt:
            return TRADING_HOURS_PER_DAY
        return (dt - open_dt).total_seconds() / 3600.0
    elapsed = 0.0
    current = buy_time.date()
    end     = now.date()
    while current <= end:
        if current.weekday() < 5:
            if current == buy_time.date() == now.date():
                elapsed += clamp_to_market(now) - clamp_to_market(buy_time)
            elif current == buy_time.date():
                elapsed += TRADING_HOURS_PER_DAY - clamp_to_market(buy_time)
            elif current == end:
                elapsed += clamp_to_market(now)
            else:
                elapsed += TRADING_HOURS_PER_DAY
        current += timedelta(days=1)
    return max(elapsed, 0.0)

def SQLConnect() -> my.MySQLConnection | None:
    try:
        conn = my.connect(host=os.getenv("DATABASE_HOST"), user=os.getenv("DATABASE_USERNAME"),
                          password=os.getenv("DATABASE_PASSWORD"),database="ALGOTRADER")
        return conn if conn.is_connected() else None
    except Exception as e:
        logging.error(f"DB connect failed: {e}")
        return None

def SQLClose(conn, cursor) -> None:
    if cursor:
        cursor.close()
    if conn and conn.is_connected():
        conn.close()

def LogTrade(conn, cursor, ticker: str, side: str, quantity: float, price: float, total: float, PORTFOLIO: float = PORTFOLIO) -> None:
    try:
        QUERY  = "INSERT INTO TRADES (TICKER, SIDE, QUANTITY, PRICE, TOTAL, PORTFOLIO) VALUES (%s, %s, %s, %s, %s, %s)"
        VALUES = (ticker, side.lower(), round(quantity, 6), round(price, 4), round(total, 4), round(PORTFOLIO, 4))
        cursor.execute(QUERY, VALUES)
        conn.commit()
    except Exception as e:
        logging.error(f"LogTrade failed: {e}")

def Login(username = USERNAME, password = PASSWORD) -> None:
    try:
        rh.login(username, password)
        logging.info("Logged in to Robinhood.")
    except Exception as e:
        logging.error(f"Login failed: {e}")
        exit(1)

def Logout() -> None:
    rh.authentication.logout()
    logging.info("Logged out of Robinhood.")

def GetLiveFeatures(ticker: str) -> np.ndarray | None:
    logging.info(f"Downloading Features: {ticker}")
    data = rh.stocks.get_stock_historicals(ticker, interval='hour', span='3month', bounds='regular', info=None)
    if not data or len(data) < 50:
        logging.warning(f"Insufficient data: {ticker}")
        return None

    closes = np.array([float(hour['close_price']) for hour in data])
    highs = np.array([float(hour['high_price'])  for hour in data])
    lows = np.array([float(hour['low_price']) for hour in data])
    volumes = np.array([float(hour['volume']) for hour in data])
    index = len(closes) - 1

    rsi = CalculateRSI(list(closes[:index+1]))
    change_1 = SafeDivide(closes[index] - closes[index-1],  closes[index-1])
    change_5 = SafeDivide(closes[index] - closes[index-5],  closes[index-5])
    change_20 = SafeDivide(closes[index] - closes[index-20], closes[index-20])
    ma20_ratio = SafeDivide(closes[index], np.mean(closes[index-20:index]))
    ma50_ratio = SafeDivide(closes[index], np.mean(closes[index-50:index]))
    volatility_14 = Volatility(closes[index-14:index])
    volatility_20 = Volatility(closes[index-20:index])
    vol_ratio = SafeDivide(volumes[index], np.mean(volumes[index-20:index]))
    high_low = SafeDivide(highs[index] - lows[index], lows[index])
    macd, macd_signal, macd_hist = CalculateMACD(closes[:index+1])
    band_pos = CalculateBollinger(closes[:index+1])
    intraday_pos = SafeDivide(closes[index] - lows[index], highs[index] - lows[index])
    ticker_idx = WATCHLIST.index(ticker) / len(WATCHLIST)
    log_avg_vol = math.log(np.mean(volumes[index-20:index]) + 1)
    atr = CalculateATR(highs[index-14:index+1], lows[index-14:index+1], closes[index-14:index+1])

    return np.array([[rsi, change_1, change_5, change_20, ma20_ratio,
                      ma50_ratio, volatility_14, volatility_20, vol_ratio, high_low,
                      macd, macd_signal, macd_hist, band_pos, intraday_pos,ticker_idx, 
                      log_avg_vol, atr]], dtype=float)

def BuyOrder(conn, cursor, ticker: str, spend: float, confidence: float = 0.0) -> bool:
    global CAPITAL, PORTFOLIO
    try:
        current_price = LivePrice(ticker)
        quantity  = spend / current_price
        HOLDINGS[ticker] = quantity
        PURCHASE_PRICES[ticker] = current_price
        BUY_TIMESTAMPS[ticker] = datetime.now()
        CAPITAL -= spend
        UpdatePortfolio()
        LogTrade(conn, cursor, ticker, "buy", quantity, current_price, spend, PORTFOLIO)
        message = (f"BUY {ticker}: {quantity:.4f} shares @ ${current_price:.2f} | Spent ${spend:.2f} | Confidence {confidence:.2%}")
        logging.info(message)
        return True
    except Exception as e:
        logging.error(f"BuyOrder failed for {ticker}: {e}")
        return False

def SellOrder(conn, cursor, ticker: str, reason: str = "SIGNAL") -> bool:
    global CAPITAL, PORTFOLIO
    try:
        quantity = HOLDINGS[ticker]
        if quantity <= 0:
            return False
        current_price = LivePrice(ticker)
        proceeds = quantity * current_price
        pnl = proceeds - (quantity * PURCHASE_PRICES[ticker])
        HOLDINGS[ticker] = 0.0
        PURCHASE_PRICES[ticker] = 0.0
        BUY_TIMESTAMPS[ticker] = None
        CAPITAL += proceeds
        UpdatePortfolio()
        LogTrade(conn, cursor, ticker, "sell", quantity, current_price, proceeds, PORTFOLIO)
        message = (f"SELL {ticker} [{reason}]: {quantity:.4f} shares @ ${current_price:.2f} | Proceeds ${proceeds:.2f} | PnL ${pnl:.2f}")
        logging.info(message)
        return True
    except Exception as e:
        logging.error(f"SellOrder failed for {ticker}: {e}")
        return False

def LivePrice(ticker: str) -> float:
    price = rh.stocks.get_latest_price(ticker, priceType=None, includeExtendedHours=True)[0]
    return float(price)

def UpdatePortfolio() -> None:
    global PORTFOLIO, CAPITAL
    PORTFOLIO = CAPITAL
    for ticker, quantity in HOLDINGS.items():
        if quantity > 0:
            price = LivePrice(ticker)
            PORTFOLIO += quantity * price

def main():
    Login()
    conn = SQLConnect()
    if conn and conn.is_connected():
        logging.info("Successfully Connected to Database")
    cursor = conn.cursor()
    logging.info("=== Algo Trader Started ===")

    try:
        while True:
            if not conn or not conn.is_connected():
                conn = SQLConnect()
                cursor = conn.cursor() if conn else None

            if not MarketHours():
                secs = SecondsUntilMarketOpen()
                hrs  = secs / 3600
                logging.info(f"Market closed. Next open in {hrs:.1f}h — sleeping.")
                time.sleep(min(secs, 1800))
                continue

            now = datetime.now()
            logging.info(f"[{now.strftime('%H:%M:%S')}] Scanning...")

            for ticker in WATCHLIST:
                if HOLDINGS[ticker] > 0:
                    current_price = LivePrice(ticker)
                    returns = (current_price - PURCHASE_PRICES[ticker]) / PURCHASE_PRICES[ticker]
                    hours_held = MarketMinutesElapsed(BUY_TIMESTAMPS[ticker], datetime.now())

                    if hours_held >= HOLD_HOURS:
                        SellOrder(conn, cursor, ticker, reason="5H_HOLD")
                    elif returns >= TAKE_PROFIT:
                        SellOrder(conn, cursor, ticker, reason="TAKE_PROFIT")
                    elif returns <= -STOP_LOSS:
                        SellOrder(conn, cursor, ticker, reason="STOP_LOSS")

            open_positions = sum([1 for v in HOLDINGS.values() if v > 0])
            if open_positions < MAX_POSITIONS:
                predictions = dict()
                for ticker in WATCHLIST:
                    if HOLDINGS[ticker] == 0:
                        features = GetLiveFeatures(ticker)
                        if features is None:
                            continue
                        features_scaled = scaler.transform(features)
                        predictions[ticker] = float(MODEL.predict(features_scaled).flatten()[0])

                ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                for ticker, confidence in ranked:
                    if open_positions >= MAX_POSITIONS:
                        break
                    if confidence > 0.55:
                        spend = POSITION_SIZE
                        if spend > CAPITAL:
                            continue
                        if BuyOrder(conn, cursor, ticker, spend, confidence):
                            open_positions += 1
                            NO_OF_TRADES.append(NO_OF_TRADES[-1] + 1)
                            PORTFOLIO_CHANGE.append(PORTFOLIO)
                            with open("equitycurve.txt", "a") as file:
                                file.write(f"{NO_OF_TRADES[-1]},{PORTFOLIO}\n")
            
            UpdatePortfolio()
            logging.info(f"Capital: ${CAPITAL:,.2f} | Open positions: {open_positions} | Portfolio: ${PORTFOLIO:,.2f}")
            time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")

    finally:
        logging.info(f"=== Session Complete | Final capital: ${CAPITAL:,.2f} | Final Portfolio: ${PORTFOLIO:,.2f} ===")
        Logout()
        SQLClose(conn, cursor)

if __name__ == "__main__":
    main()