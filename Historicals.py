import os
from dotenv import load_dotenv
import robin_stocks.robinhood as rh
from trader import LOGIN, LOGOUT, WATCHLIST
from ModularNeuralNetwork import ModularNeuralNet
import numpy as np
from statistics import mean, stdev
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import math

load_dotenv("credentials.env")
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD") 

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
        return 0.01
    return a / b

def CalculateRSI(prices: list[float], period: int = 14) -> float | None:
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

def Volatility(prices: list[float]) -> float | None:
    log_ratio = []
    for i in range(1, len(prices)):
        log_ratio.append(math.log(SafeDivide(prices[i], prices[i-1])))
    return stdev(log_ratio)

def BuildTrainingData(tickers: list[str] = WATCHLIST) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    X, Y, prices, timestamps, future, = [], [], [], [], []

    for ticker in tickers:
        print(f"Downloading Historical Data: {ticker}")
        data = rh.stocks.get_stock_historicals(
            ticker, interval='hour', span='3month', bounds='regular', info=None
        )
        if not data or len(data) < 60:
            print(f"Download Unsuccessful: {ticker}")
            continue

        closes  = np.array([float(d['close_price']) for d in data])
        highs   = np.array([float(d['high_price'])  for d in data])
        lows    = np.array([float(d['low_price'])   for d in data])
        volumes = np.array([float(d['volume'])      for d in data])
        times   = [d['begins_at'] for d in data]

        for i in range(50, len(closes) - 5):
            rsi        = CalculateRSI(list(closes[:i+1]))
            change_1   = SafeDivide(closes[i] - closes[i-1],  closes[i-1])
            change_5   = SafeDivide(closes[i] - closes[i-5],  closes[i-5])
            change_20  = SafeDivide(closes[i] - closes[i-20], closes[i-20])
            ma20_ratio = SafeDivide(closes[i], np.mean(closes[i-20:i]))
            ma50_ratio = SafeDivide(closes[i], np.mean(closes[i-50:i]))
            volatility_14 = Volatility(closes[i-14:i])
            volatility_20 = Volatility(closes[i-20:i])
            vol_ratio  = SafeDivide(volumes[i], np.mean(volumes[i-20:i]))
            high_low   = SafeDivide(highs[i] - lows[i], lows[i])
            macd, macd_signal, macd_hist = CalculateMACD(closes[:i+1])
            band_pos = CalculateBollinger(closes[:i+1])
            # dt = datetime.fromisoformat(times[i].replace("Z", "+00:00"))
            # hour = dt.hour
            # hour_sin = math.sin(2 * math.pi * hour / 24)
            # hour_cos = math.cos(2 * math.pi * hour / 24)
            intraday_pos = SafeDivide(closes[i] - lows[i], highs[i] - lows[i])

            future_return = SafeDivide(closes[i+5] - closes[i], closes[i])
            label = 1 if future_return > 0.01 else (0 if future_return < -0.01 else None)
            if label is None:
                continue

            future.append(future_return)
            X.append([rsi, change_1, change_5, change_20, ma20_ratio,
                      ma50_ratio, volatility_14, volatility_20, vol_ratio, high_low,
                      macd, macd_signal, macd_hist, band_pos, intraday_pos])
            Y.append(label)
            prices.append(closes[i])
            timestamps.append(times[i])

    if not X:
        return None

    sort_order = np.argsort(timestamps)
    X      = np.array(X, dtype=float)[sort_order]
    Y      = np.array(Y, dtype=float)[sort_order]
    prices = np.array(prices, dtype=float)[sort_order]
    future = [future[i] for i in sort_order]
    timestamps = [timestamps[i] for i in sort_order]

    return X, Y, timestamps, future

X, Y, timestamps, future = BuildTrainingData(WATCHLIST)
print(f"Total Sample Size: {X.size}")
scaler = StandardScaler()
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
joblib.dump(scaler, "scaler.save")
_, future_val = train_test_split(future, test_size=0.2, shuffle=False)
_, timestamps_val = train_test_split(timestamps, test_size=0.2, shuffle=False)
model = ModularNeuralNet(input_size=15, hidden_layers=[64, 32, 16, 8, 1],
                          activation='relu', final_activation='sigmoid')

idx_0 = np.where(Y_train == 0)[0]
idx_1 = np.where(Y_train == 1)[0]
num_to_keep = len(idx_1)
idx_0_sampled = np.random.choice(idx_0, num_to_keep, replace=False)
balanced_indices = np.concatenate([idx_0_sampled, idx_1])
np.random.shuffle(balanced_indices)

X_train_balanced = X_train[balanced_indices]
Y_train_balanced = Y_train[balanced_indices] 

print(f"Train Sample Size: {X_train_balanced.size}")
print(f"Validation Sample Size: {X_val.size}")
print(f"Validation Period: {Y_val.size}")
print(np.unique(Y_train_balanced, return_counts=True))
unique_hours = len(set(timestamps_val))
trading_days = unique_hours / 6.5
print(f"Unique hours:    {unique_hours}")
print(f"Trading days:    {trading_days:.0f}")
print(f"Trading months:  {trading_days / 21:.1f}")

model.train(X_train_balanced, Y_train_balanced, epochs=55000, learning_rate=0.0005,
    batch_size=64, learning_rate_decay=0.999, decay_interval=50,
    validation_data=(X_val, Y_val), early_stopping_patience=750,
    print_interval=100)

metrics, _ = model.evaluate(X_val, Y_val)
print(metrics)
model.save_model("trader_model.npy")

LOGOUT()