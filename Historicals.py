import os
from dotenv import load_dotenv
import robin_stocks.robinhood as rh
from trader import LOGIN, LOGOUT, WATCHLIST
import numpy as np
from statistics import mean



load_dotenv("credentials.env") # Load environment variables from .env file
USERNAME = os.getenv("USERNAME") # Set your Robinhood username as an environment variable
PASSWORD = os.getenv("PASSWORD") # Set your Robinhood password as an environment variable

LOGIN(username=USERNAME, password=PASSWORD)

def CalculateRSI(prices: list[float], period: int = 14) -> float | None:
    """
    Classic Wilder RSI.
    Requires at least period + 1 data points.
    Returns None if there isn't enough data.
    """
    if len(prices) < period + 1:
        return None

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    seed_deltas = deltas[:period]
    avg_gain = mean([d for d in seed_deltas if d > 0] or [0])
    avg_loss = mean([-d for d in seed_deltas if d < 0] or [0])

    for delta in deltas[period:]:
        gain = delta if delta > 0 else 0
        loss = -delta if delta < 0 else 0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def BuildTrainingData(tickers: list[str] = WATCHLIST) -> tuple[np.array, np.array] | None:
    X, Y =[], []

    for ticker in tickers:
        data = rh.stocks.get_stock_historicals(ticker, interval='day', span='5year', bounds='regular', info=None)
        if (not data or len(data) < 60): 
            continue
        
        closes = np.array([float(hour['close_price']) for hour in data])
        highs = np.array([float(hour['high_price']) for hour in data])
        lows = np.array([float(hour['low_price']) for hour in data])
        volumes = np.array([float(hour['volume']) for hour in data])

        for i in range(50, len(closes) - 1):
            window_closes  = closes[:i+1]

            rsi         = CalculateRSI(list(window_closes.astype(float)))
            change_1    = (closes[i] - closes[i-1])  / closes[i-1]
            change_5    = (closes[i] - closes[i-5])  / closes[i-5]
            change_20   = (closes[i] - closes[i-20]) / closes[i-20]
            ma50_ratio  = closes[i] / np.mean(closes[i-50:i])
            volatility  = np.std(closes[i-14:i])
            vol_ratio   = volumes[i] / np.mean(volumes[i-20:i])
            high_low    = (highs[i] - lows[i]) / lows[i]

            label = 1 if closes[i+1] > closes[i] else 0

            X.append([rsi, change_1, change_5, change_20, ma50_ratio, volatility, vol_ratio, high_low])
            Y.append(label)

    return np.array(X), np.array(Y)

LOGOUT()