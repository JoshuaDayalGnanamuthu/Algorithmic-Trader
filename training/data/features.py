import numpy as np
import math
import pandas as pd
from statistics import mean, stdev
from utils.math_utils import SafeDivide
from config import DATA_CONFIG, LABEL_CONFIG


def CalculateRSI(prices: list[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0

    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    seed   = deltas[:period]
    avg_gain = mean([d for d in seed if d > 0] or [0])
    avg_loss = mean([-d for d in seed if d < 0] or [0])

    for delta in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(delta, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-delta, 0)) / period

    return 100.0 if avg_loss == 0 else round(100 - (100 / (1 + avg_gain / avg_loss)), 2)


def CalculateMACD(closes, fast=12, slow=26, signal=9) -> tuple[float, float, float]:
    s         = pd.Series(closes)
    macd_line = s.ewm(span=fast).mean() - s.ewm(span=slow).mean()
    sig_line  = macd_line.ewm(span=signal).mean()
    return float(macd_line.iloc[-1]), float(sig_line.iloc[-1]), float((macd_line - sig_line).iloc[-1])


def CalculateBollinger(closes, period=20) -> float:
    s   = pd.Series(closes)
    ma  = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = float((ma + 2 * std).iloc[-1])
    lower = float((ma - 2 * std).iloc[-1])
    return SafeDivide(closes[-1] - lower, upper - lower + 1e-10)


def CalculateATR(highs, lows, closes, period=14) -> float:
    trs = [
        max(highs[j] - lows[j],
            abs(highs[j] - closes[j-1]),
            abs(lows[j]  - closes[j-1]))
        for j in range(1, len(closes))
    ]
    return SafeDivide(np.mean(trs[-period:]), closes[-1])


def Volatility(prices: list[float]) -> float:
    logs = [math.log(SafeDivide(prices[i], prices[i-1])) for i in range(1, len(prices))]
    return stdev(logs) if len(logs) > 1 else 0.0


def BuildFeatureVector(closes, highs, lows, volumes, i, ticker_idx) -> list[float]:
    return [
        CalculateRSI(list(closes[:i+1])),
        SafeDivide(closes[i] - closes[i-1],  closes[i-1]),
        SafeDivide(closes[i] - closes[i-5],  closes[i-5]),
        SafeDivide(closes[i] - closes[i-20], closes[i-20]),
        SafeDivide(closes[i], np.mean(closes[i-20:i])),
        SafeDivide(closes[i], np.mean(closes[i-50:i])),
        Volatility(closes[i-14:i]),
        Volatility(closes[i-20:i]),
        SafeDivide(volumes[i], np.mean(volumes[i-20:i])),
        SafeDivide(highs[i] - lows[i], lows[i]),
        *CalculateMACD(closes[:i+1]),
        CalculateBollinger(closes[:i+1]),
        SafeDivide(closes[i] - lows[i], highs[i] - lows[i]),
        ticker_idx,
        math.log(np.mean(volumes[i-20:i]) + 1),
        CalculateATR(highs[i-14:i+1], lows[i-14:i+1], closes[i-14:i+1]),
    ]


def BuildLabel(closes, i) -> int | None:
    future_return = SafeDivide(closes[i+5] - closes[i], closes[i])
    if future_return >  LABEL_CONFIG["buy_threshold"]:  return 1, future_return
    if future_return < LABEL_CONFIG["sell_threshold"]:  return 0, future_return
    return None, future_return


def BuildDataset(raw_data: dict, watchlist: list[str]) -> tuple:
    """
    raw_data: output of FetchAll — { ticker: { closes, highs, lows, volumes, times } }
    Returns: X, Y, timestamps, future_returns (all sorted chronologically)
    """
    X, Y, prices, timestamps, future = [], [], [], [], []

    for ticker, data in raw_data.items():
        closes  = data["closes"]
        highs   = data["highs"]
        lows    = data["lows"]
        volumes = data["volumes"]
        times   = data["times"]

        ticker_idx = watchlist.index(ticker) / len(watchlist)
        warmup     = DATA_CONFIG["warmup"]
        forward    = DATA_CONFIG["forward_bars"]

        for i in range(warmup, len(closes) - forward):
            label, future_return = BuildLabel(closes, i)
            if label is None:
                continue

            features = BuildFeatureVector(closes, highs, lows, volumes, i, ticker_idx)

            X.append(features)
            Y.append(label)
            future.append(future_return)
            prices.append(closes[i])
            timestamps.append(times[i])

    if not X:
        return None

    order      = np.argsort(timestamps)
    X          = np.array(X, dtype=float)[order]
    Y          = np.array(Y, dtype=float)[order]
    timestamps = [timestamps[i] for i in order]
    future     = [future[i]     for i in order]

    print(f"Dataset built — {len(X)} samples across {len(raw_data)} tickers")
    return X, Y, timestamps, future