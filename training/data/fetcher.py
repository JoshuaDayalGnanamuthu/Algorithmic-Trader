import robin_stocks.robinhood as rh
import numpy as np

def FetchTicker(ticker: str, config: dict) -> dict | None:
    data = rh.stocks.get_stock_historicals(
        ticker,
        interval=config["interval"],
        span=config["span"],
        bounds=config["bounds"],
        info=None
    )
    if not data or len(data) < config["min_bars"]:
        print(f"Download Unsuccessful: {ticker}")
        return None

    return {
        "closes": np.array([float(hour['close_price']) for hour in data]),
        "highs": np.array([float(hour['high_price']) for hour in data]),
        "lows": np.array([float(hour['low_price']) for hour in data]),
        "volumes": np.array([float(hour['volume']) for hour in data]),
        "times": [hour['begins_at'] for hour in data],
    }

def FetchAll(tickers: list[str], config: dict) -> dict:
    results = {}
    for ticker in tickers:
        print(f"Downloading: {ticker}")
        data = FetchTicker(ticker, config)
        if data:
            results[ticker] = data
    return results