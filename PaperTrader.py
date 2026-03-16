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
from Historicals import CalculateRSI, Volatility, CalculateBollinger, CalculateMACD, SafeDivide
import joblib

load_dotenv("credentials.env")
USERNAME    = os.getenv("USERNAME")
PASSWORD    = os.getenv("PASSWORD")
MODEL       = ModularNeuralNet.load_model(r'/Users/joshuadayal/Documents/Python/Algorithmic-Trader/files/trader_model.npy')
PRIVATE_KEY = RSA.import_key(os.getenv("PRIVATE_KEY").replace("\\n", "\n"))
PUBLIC_KEY  = RSA.import_key(os.getenv("PUBLIC_KEY").replace("\\n", "\n"))
CAPITAL     = 100000.0

WATCHLIST = [
    "AAPL", "TSLA", "ASTS", "NVDA", "AMZN",
    "MSFT", "GOOGL", "META", "AMD", "INTC",
    "RIVN", "RKLB", "SPY", "QQQ"
]

HOLD_HOURS      = 5
HOLDINGS        = dict.fromkeys(WATCHLIST, 0.0)
PURCHASE_PRICES = dict.fromkeys(WATCHLIST, 0.0)
BUY_TIMESTAMPS  = dict.fromkeys(WATCHLIST, None)

SCAN_INTERVAL   = 300       # seconds between scans during market hours
AFTER_HOURS_SLEEP = 60      # seconds between checks during after hours
TAKE_PROFIT     = 0.03
STOP_LOSS       = 0.005
POSITION_SIZE   = 0.05
MAX_POSITIONS   = 10

scaler = joblib.load(r"files/scaler.save")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trader.log"),
        logging.StreamHandler()
    ]
)

def MarketHours() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.hour * 100 + now.minute
    return 930 <= t < 1600

def SecondsUntilMarketOpen() -> float:
    """Return seconds until next market open (9:30 on a weekday)."""
    now  = datetime.now()
    days_ahead = 0
    while True:
        candidate = (now + timedelta(days=days_ahead)).replace(
            hour=9, minute=30, second=0, microsecond=0
        )
        if candidate > now and candidate.weekday() < 5:
            return (candidate - now).total_seconds()
        days_ahead += 1

def MarketMinutesElapsed(buy_time: datetime, now: datetime) -> float:
    elapsed = 0.0
    cursor  = buy_time

    while cursor < now:
        if cursor.weekday() >= 5:
            cursor += timedelta(hours=1)
            continue

        market_open  = cursor.replace(hour=9,  minute=30, second=0, microsecond=0)
        market_close = cursor.replace(hour=16, minute=0,  second=0, microsecond=0)

        if cursor < market_open:
            cursor = market_open
            continue

        if cursor >= market_close:
            next_day = (cursor + timedelta(days=1)).replace(
                hour=9, minute=30, second=0, microsecond=0
            )
            cursor = next_day
            continue

        next_minute = cursor + timedelta(minutes=1)
        end         = min(next_minute, now)
        elapsed    += (end - cursor).seconds / 3600.0
        cursor      = next_minute

    return elapsed

# ── db ────────────────────────────────────────────────────────────────────────

def SQLConnect() -> my.MySQLConnection | None:
    try:
        conn = my.connect(
            host=os.getenv("DATABASE_HOST"),
            user=os.getenv("DATABASE_USERNAME"),
            password=os.getenv("DATABASE_PASSWORD"),
            database="ALGOTRADER"
        )
        return conn if conn.is_connected() else None
    except Exception as e:
        logging.error(f"DB connect failed: {e}")
        return None

def SQLClose(conn, cursor) -> None:
    if cursor:
        cursor.close()
    if conn and conn.is_connected():
        conn.close()

def LogTrade(conn, cursor, ticker: str, side: str, quantity: float,
             price: float, total: float, message: str) -> None:
    try:
        QUERY  = "INSERT INTO TRADES (TICKER, SIDE, QUANTITY, PRICE, TOTAL) VALUES (%s, %s, %s, %s, %s)"
        VALUES = (ticker, side.lower(), round(quantity, 6), round(price, 4), round(total, 4))
        cursor.execute(QUERY, VALUES)
        conn.commit()
    except Exception as e:
        logging.error(f"LogTrade failed: {e}")

def Login() -> None:
    try:
        rh.login(USERNAME, PASSWORD)
        logging.info("Logged in to Robinhood.")
    except Exception as e:
        logging.error(f"Login failed: {e}")
        exit(1)

def Logout() -> None:
    rh.authentication.logout()
    logging.info("Logged out of Robinhood.")

def GetLiveFeatures(ticker: str) -> np.ndarray | None:
    logging.info(f"Fetching features: {ticker}")
    data = rh.stocks.get_stock_historicals(
        ticker, interval='hour', span='3month', bounds='regular', info=None
    )
    if not data or len(data) < 50:
        logging.warning(f"Insufficient data: {ticker}")
        return None

    closes  = np.array([float(h['close_price']) for h in data])
    highs   = np.array([float(h['high_price'])  for h in data])
    lows    = np.array([float(h['low_price'])   for h in data])
    volumes = np.array([float(h['volume'])      for h in data])
    i = len(closes) - 1

    rsi           = CalculateRSI(list(closes[:i+1]))
    change_1      = SafeDivide(closes[i] - closes[i-1],  closes[i-1])
    change_5      = SafeDivide(closes[i] - closes[i-5],  closes[i-5])
    change_20     = SafeDivide(closes[i] - closes[i-20], closes[i-20])
    ma20_ratio    = SafeDivide(closes[i], np.mean(closes[i-20:i]))
    ma50_ratio    = SafeDivide(closes[i], np.mean(closes[i-50:i]))
    volatility_14 = Volatility(closes[i-14:i])
    volatility_20 = Volatility(closes[i-20:i])
    vol_ratio     = SafeDivide(volumes[i], np.mean(volumes[i-20:i]))
    high_low      = SafeDivide(highs[i] - lows[i], lows[i])
    macd, macd_signal, macd_hist = CalculateMACD(closes[:i+1])
    band_pos      = CalculateBollinger(closes[:i+1])
    intraday_pos  = SafeDivide(closes[i] - lows[i], highs[i] - lows[i])

    return np.array([[rsi, change_1, change_5, change_20, ma20_ratio,
                      ma50_ratio, volatility_14, volatility_20, vol_ratio, high_low,
                      macd, macd_signal, macd_hist, band_pos, intraday_pos]], dtype=float)

def BuyOrder(conn, cursor, ticker: str, spend: float, confidence: float = 0.0) -> bool:
    global CAPITAL
    try:
        current_price           = float(rh.stocks.get_latest_price(ticker, priceType=None, includeExtendedHours=False)[0])
        quantity                = spend / current_price
        HOLDINGS[ticker]        = quantity
        PURCHASE_PRICES[ticker] = current_price
        BUY_TIMESTAMPS[ticker]  = datetime.now()
        CAPITAL                -= spend

        message = (f"BUY {ticker}: {quantity:.4f} shares @ ${current_price:.2f} "
                   f"| Spent ${spend:.2f} | Confidence {confidence:.2%}")
        LogTrade(conn, cursor, ticker, "BUY", quantity, current_price, spend, message)
        logging.info(message)
        return True

    except Exception as e:
        logging.error(f"BuyOrder failed for {ticker}: {e}")
        return False

def SellOrder(conn, cursor, ticker: str, reason: str = "SIGNAL") -> bool:
    global CAPITAL
    try:
        quantity = HOLDINGS[ticker]
        if quantity <= 0:
            return False

        current_price           = float(rh.stocks.get_latest_price(ticker, priceType=None, includeExtendedHours=False)[0])
        proceeds                = quantity * current_price
        pnl                     = proceeds - (quantity * PURCHASE_PRICES[ticker])
        HOLDINGS[ticker]        = 0.0
        PURCHASE_PRICES[ticker] = 0.0
        BUY_TIMESTAMPS[ticker]  = None
        CAPITAL                += proceeds

        message = (f"SELL {ticker} [{reason}]: {quantity:.4f} shares @ ${current_price:.2f} "
                   f"| Proceeds ${proceeds:.2f} | PnL ${pnl:.2f}")
        LogTrade(conn, cursor, ticker, "SELL", quantity, current_price, proceeds, message)
        logging.info(message)
        return True

    except Exception as e:
        logging.error(f"SellOrder failed for {ticker}: {e}")
        return False

def main():
    Login()
    conn   = SQLConnect()
    cursor = conn.cursor() if conn else None
    logging.info("=== Algo Trader Started ===")

    try:
        while True:
            if not MarketHours():
                secs = SecondsUntilMarketOpen()
                hrs  = secs / 3600
                logging.info(f"Market closed. Next open in {hrs:.1f}h — sleeping.")
                time.sleep(min(secs, 3600))   # wake up at least every hour to recheck
                continue

            now = datetime.now()
            logging.info(f"[{now.strftime('%H:%M:%S')}] Scanning...")

            for ticker in WATCHLIST:
                if HOLDINGS[ticker] > 0:
                    current_price = float(rh.stocks.get_latest_price(
                        ticker, priceType=None, includeExtendedHours=False)[0])
                    ret        = (current_price - PURCHASE_PRICES[ticker]) / PURCHASE_PRICES[ticker]
                    hours_held = MarketMinutesElapsed(BUY_TIMESTAMPS[ticker], datetime.now())

                    if hours_held >= HOLD_HOURS:
                        SellOrder(conn, cursor, ticker, reason="5H_HOLD")
                    elif ret >= TAKE_PROFIT:
                        SellOrder(conn, cursor, ticker, reason="TAKE_PROFIT")
                    elif ret <= -STOP_LOSS:
                        SellOrder(conn, cursor, ticker, reason="STOP_LOSS")

            open_positions = sum(1 for v in HOLDINGS.values() if v > 0)

            if open_positions < MAX_POSITIONS:
                predictions = {}
                for ticker in WATCHLIST:
                    if HOLDINGS[ticker] == 0:
                        features = GetLiveFeatures(ticker)
                        if features is None:
                            continue
                        features_scaled     = scaler.transform(features)
                        predictions[ticker] = float(MODEL.predict(features_scaled).flatten()[0])

                ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

                for ticker, confidence in ranked:
                    if open_positions >= MAX_POSITIONS:
                        break
                    if confidence > 0.65:
                        spend = POSITION_SIZE * CAPITAL
                        if spend > CAPITAL:
                            continue
                        if BuyOrder(conn, cursor, ticker, spend, confidence):
                            open_positions += 1

            logging.info(f"Capital: ${CAPITAL:,.2f} | Open positions: {open_positions}")
            time.sleep(SCAN_INTERVAL)

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")

    finally:
        logging.info(f"=== Session Complete | Final capital: ${CAPITAL:,.2f} ===")
        Logout()
        SQLClose(conn, cursor)

if __name__ == "__main__":
    main()