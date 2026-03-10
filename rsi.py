import robin_stocks.robinhood as rh
from datetime import datetime
from statistics import mean
from colorama import Fore, Style, init

def CalculateRSI(prices: list[float], period: int = 14) -> float | None:
    """
    Classic Wilder RSI.
    Requires at least period + 1 data points.
    Returns None if there isn't enough data.
    """
    if len(prices) < period + 1:
        return None

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains  = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]

    # Seed averages from first `period` deltas
    seed_deltas = deltas[:period]
    avg_gain = mean([d for d in seed_deltas if d > 0] or [0])
    avg_loss = mean([-d for d in seed_deltas if d < 0] or [0])

    # Wilder smoothing over remaining deltas
    for delta in deltas[period:]:
        gain = delta if delta > 0 else 0
        loss = -delta if delta < 0 else 0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def HistoricalPrices(symbol: str, span: str = "month", interval: str = "hour") -> list[float]:
    """Fetch closing prices from Robinhood historical data."""
    try:
        historicals = rh.stocks.get_stock_historicals(
            symbol, interval=interval, span=span, bounds="regular"
        )
        if not historicals:
            return []
        return [float(bar["close_price"]) for bar in historicals if bar.get("close_price")]
    except Exception as e:
        log.error(f"Failed to fetch historicals for {symbol}: {e}")
        return []

def CurrentPrices(symbol: str) -> float | None:
    """Get the latest trade price."""
    try:
        price = rh.stocks.get_latest_price(symbol)
        return float(price[0]) if price else None
    except Exception as e:
        log.error(f"Failed to fetch price for {symbol}: {e}")
        return None

from datetime import datetime


init(autoreset=True)

def AlertEngine(symbol: str, rsi: float, price: float, signal: str) -> str:
    """
    Generate a formatted trading alert message.
    """
    if rsi < 30:
        zone = "OVERSOLD"
    elif rsi > 70:
        zone = "OVERBOUGHT"
    else:
        zone = "NEUTRAL"

    # Determine signal formatting
    if signal == "BUY":
        color = Fore.GREEN
        emoji = "🟢"
    else:
        color = Fore.RED
        emoji = "🔴"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    alert = (
        f"\n"
        f"{color}{emoji} TRADEPILOT SIGNAL DETECTED{Style.RESET_ALL}\n"
        f"{'━' * 36}\n"
        f" Symbol      : {symbol}\n"
        f" Signal      : {color}{signal}{Style.RESET_ALL}\n"
        f"\n"
        f" RSI Value   : {rsi:.2f}\n"
        f" RSI Zone    : {zone}\n"
        f"\n"
        f" Price       : ${price:.2f}\n"
        f"\n"
        f" Timestamp   : {timestamp}\n"
        f"{'━' * 36}\n"
    )

    return alert