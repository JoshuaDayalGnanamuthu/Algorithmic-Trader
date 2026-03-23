from statistics import mean, stdev
import pandas as pd
import numpy as np
import math


def SafeDivide(a, b) -> float:
    if b == 0 or np.isnan(b) or np.isinf(b):
        return 0.01
    return a / b


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
