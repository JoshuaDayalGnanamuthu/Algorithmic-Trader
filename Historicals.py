import os
from dotenv import load_dotenv
import robin_stocks.robinhood as rh
from trader import LOGIN, LOGOUT, WATCHLIST
from ModularNeuralNetwork import ModularNeuralNet
import numpy as np
from statistics import mean
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

load_dotenv("credentials.env") # Load environment variables from .env file
USERNAME = os.getenv("USERNAME") # Set your Robinhood username as an environment variable
PASSWORD = os.getenv("PASSWORD") # Set your Robinhood password as an environment variable

LOGIN(username=USERNAME, password=PASSWORD)

def CalculateMACD(closes, fast=12, slow=26, signal=9):
    ema_fast   = pd.Series(closes).ewm(span=fast).mean()
    ema_slow   = pd.Series(closes).ewm(span=slow).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram  = macd_line - signal_line
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])

def CalculateBollinger(closes, period=20):
    closes_series = pd.Series(closes)
    ma  = closes_series.rolling(period).mean()
    std = closes_series.rolling(period).std()
    upper = ma + (2 * std)
    lower = ma - (2 * std)
    band_position = (closes[-1] - float(lower.iloc[-1])) / (float(upper.iloc[-1]) - float(lower.iloc[-1]) + 1e-10)
    return band_position

def SafeDivide(a, b) -> float:
    if b == 0 or np.isnan(b) or np.isinf(b):
        return 0.0
    return a / b

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
        data = rh.stocks.get_stock_historicals(ticker, interval='hour', span='3month', bounds='regular', info=None)
        if (not data or len(data) < 60): 
            continue
        
        closes = np.array([float(hour['close_price']) for hour in data])
        highs = np.array([float(hour['high_price']) for hour in data])
        lows = np.array([float(hour['low_price']) for hour in data])
        volumes = np.array([float(hour['volume']) for hour in data])

        for i in range(50, len(closes) - 5):
            window_closes  = closes[:i+1]

            rsi         = CalculateRSI(list(window_closes.astype(float)))
            change_1   = SafeDivide(closes[i] - closes[i-1],  closes[i-1])
            change_5   = SafeDivide(closes[i] - closes[i-5],  closes[i-5])
            change_20  = SafeDivide(closes[i] - closes[i-20], closes[i-20])
            ma50_ratio = SafeDivide(closes[i], np.mean(closes[i-50:i]))
            volatility  = np.std(closes[i-14:i])
            vol_ratio  = SafeDivide(volumes[i], np.mean(volumes[i-20:i]))
            high_low   = SafeDivide(highs[i] - lows[i], lows[i])
            macd, macd_signal, macd_hist = CalculateMACD(closes[:i+1])
            band_pos = CalculateBollinger(closes[:i+1])
            
            future_return = (closes[i+5] - closes[i]) / closes[i]
            label = 1 if future_return > 0.01 else (0 if future_return < -0.01 else None)
            if label is None:
                continue

            X.append([rsi, change_1, change_5, change_20,
                      ma50_ratio, volatility, vol_ratio,
                        high_low, macd, macd_signal, macd_hist,
                        band_pos])
            Y.append(label)

    return np.array(X), np.array(Y)

X, Y = BuildTrainingData(WATCHLIST)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.save")
X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y, test_size=0.2, shuffle=False)

model = ModularNeuralNet(input_size=12, hidden_layers=[64, 32, 16, 8, 1],
                          activation='relu', final_activation='sigmoid')

model.train(X_train, Y_train, epochs=55000, learning_rate=0.001,
    batch_size=64, learning_rate_decay=0.99, decay_interval=50,
    validation_data=(X_val, Y_val), early_stopping_patience=55000,
    print_interval=100)

metrics, _ = model.evaluate(X_val, Y_val)
print(metrics)
model.save_model("trader_model.npy")
LOGOUT()