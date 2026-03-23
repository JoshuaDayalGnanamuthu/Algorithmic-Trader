from ModularNeuralNetwork import ModularNeuralNet
from datetime import datetime, timedelta
import robin_stocks.robinhood as rh
from statistics import mean, stdev
import mysql.connector as my
import pandas as pd
import numpy as np
import logging
import joblib
import math
import time
import os


class PaperTrader:
    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
        self.capital = 100000.0
        self.portfolio = self.capital
        self.watchlist= [
            # Mega-cap Tech
            "AAPL", "TSLA", "ASTS", "NVDA", "AMZN",
            "MSFT", "GOOGL", "META", "AMD", "INTC",

            # AI / Semiconductors
            "AVGO",   # Broadcom
            "QCOM",   # Qualcomm
            "ARM",    # Arm Holdings
            "MRVL",   # Marvell Technology
            "SMCI",   # Super Micro Computer
            "TSM",    # Taiwan Semiconductor
            "ASML",   # ASML Holding
            "MU",     # Micron Technology

            # EV / Clean Energy
            "RIVN", "NIO", "LCID", "F", "GM",
            "CHPT",   # ChargePoint
            "BLNK",   # Blink Charging
            "ENPH",   # Enphase Energy

            # Space / Defense Tech
            "RKLB",
            "SPCE",   # Virgin Galactic
            "BWXT",   # BWX Technologies
            "LMT",    # Lockheed Martin
            "RTX",    # Raytheon

            # Growth / Software
            "CRM",    # Salesforce
            "NOW",    # ServiceNow
            "SNOW",   # Snowflake
            "PLTR",   # Palantir
            "NET",    # Cloudflare

            # ETFs
            "SPY", "QQQ",
            "SOXX",   # Semiconductor ETF
            "ARKK",   # ARK Innovation ETF
            "IWM",    # Russell 2000
            "XLK",    # Tech Sector ETF
        ]
        self.holdings = dict.fromkeys(self.watchlist, 0.0)
        self.purchase_prices = dict.fromkeys(self.watchlist, 0.0)
        self.buy_timestamps = dict.fromkeys(self.watchlist, None)
        self.model = ModularNeuralNet.load_model("files/trader_model.npy")
        self.scaler = joblib.load("files/scaler.save")
        self.hold_hours = 5
        self.conn = None
        self.cursor = None
        self.no_of_trades = [0]
        self.scan_interval = 300
        self.take_profit = 0.03
        self.stop_loss = 0.03
        self.position_size = 5000
        self.max_positions = 10
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler("trades.log"), logging.StreamHandler()])
        self.logging = logging.getLogger(__name__)
        self.market_open  = (9, 30)
        self.market_close = (16, 0)
        self.trading_hours = 6.5
    
    def SQLConnect(self) -> bool:
        try:
            self.conn = my.connect(host=os.getenv("DATABASE_HOST"), user=os.getenv("DATABASE_USERNAME"),
                            password=os.getenv("DATABASE_PASSWORD"),database="ALGOTRADER")
            if self.conn.is_connected():
                self.logging.info("Conected Successfully to MySQL")
                self.cursor = self.conn.cursor()
                return True
        except Exception as e:
            self.logging.error(f"DB connect failed: {e}")
            return False

    def SQLClose(self) -> None:
        if self.cursor:
            self.cursor.close()
        if self.conn and self.conn.is_connected():
            self.conn.close()
        self.logging.info("SQL Connection Closed.")
    
    def Login(self) -> bool:
        try:
            rh.login(self.username, self.password)
            self.logging.info("Logged in to Robinhood.")
            return True
        except Exception as e:
            self.logging.error(f"Login failed: {e}")
            return False
    
    def Logout(self) -> None:
        rh.authentication.logout()
        self.logging.info("Logged out of Robinhood.")

    def SafeDivide(self, a, b) -> float:
        if (b == 0 or np.isnan(b) or np.isinf(b)):
            return 0.0
        return a / b
    
    def CalculateMACD(self, closes, fast=12, slow=26, signal=9) -> tuple[float, float, float]:
        ema_fast = pd.Series(closes).ewm(span=fast).mean()
        ema_slow = pd.Series(closes).ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1])

    def CalculateBollinger(self, closes, period=20) -> float:
        closes_series = pd.Series(closes)
        ma  = closes_series.rolling(period).mean()
        std = closes_series.rolling(period).std()
        upper = ma + (2 * std)
        lower = ma - (2 * std)
        band_position = (closes[-1] - float(lower.iloc[-1])) / (float(upper.iloc[-1]) - float(lower.iloc[-1]) + 1e-10)
        return band_position

    def CalculateRSI(self, prices: list[float], period: int = 14) -> float | None:
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

    def Volatility(self, prices: list[float]) -> float | None:
        log_ratio = []
        for i in range(1, len(prices)):
            log_ratio.append(math.log(self.SafeDivide(prices[i], prices[i-1])))
        return stdev(log_ratio)

    def CalculateATR(self, highs: list[float], lows: list[float], closes: list[float], period=14) ->float:
        trs = []
        for j in range(1, len(closes)):
            tr = max(highs[j] - lows[j],
                    abs(highs[j] - closes[j-1]),
                    abs(lows[j] - closes[j-1]))
            trs.append(tr)
        atr = np.mean(trs[-period:])
        return self.SafeDivide(atr, closes[-1])

    def GetLiveFeatures(self, ticker: str) -> np.ndarray | None:
        self.logging.info(f"Downloading Features: {ticker}")
        data = rh.stocks.get_stock_historicals(ticker, interval='hour', span='3month', bounds='regular', info=None)

        if not data or len(data) < 50:
            self.logging.warning(f"Insufficient data: {ticker}")
            return None

        closes = np.array([float(hour['close_price']) for hour in data])
        highs = np.array([float(hour['high_price'])  for hour in data])
        lows = np.array([float(hour['low_price']) for hour in data])
        volumes = np.array([float(hour['volume']) for hour in data])
        index = len(closes) - 1

        rsi = self.CalculateRSI(list(closes[:index+1]))
        change_1 = self.SafeDivide(closes[index] - closes[index-1],  closes[index-1])
        change_5 = self.SafeDivide(closes[index] - closes[index-5],  closes[index-5])
        change_20 = self.SafeDivide(closes[index] - closes[index-20], closes[index-20])
        ma20_ratio = self.SafeDivide(closes[index], np.mean(closes[index-20:index]))
        ma50_ratio = self.SafeDivide(closes[index], np.mean(closes[index-50:index]))
        volatility_14 = self.Volatility(closes[index-14:index])
        volatility_20 = self.Volatility(closes[index-20:index])
        vol_ratio = self.SafeDivide(volumes[index], np.mean(volumes[index-20:index]))
        high_low = self.SafeDivide(highs[index] - lows[index], lows[index])
        macd, macd_signal, macd_hist = self.CalculateMACD(closes[:index+1])
        band_pos = self.CalculateBollinger(closes[:index+1])
        intraday_pos = self.SafeDivide(closes[index] - lows[index], highs[index] - lows[index])
        ticker_idx = self.watchlist.index(ticker) / len(self.watchlist)
        log_avg_vol = math.log(np.mean(volumes[index-20:index]) + 1)
        atr = self.CalculateATR(highs[index-14:index+1], lows[index-14:index+1], closes[index-14:index+1])

        return np.array([[rsi, change_1, change_5, change_20, ma20_ratio,
                        ma50_ratio, volatility_14, volatility_20, vol_ratio, high_low,
                        macd, macd_signal, macd_hist, band_pos, intraday_pos,ticker_idx, 
                        log_avg_vol, atr]], dtype=float)
    
    def LogTrade(self, ticker: str, side: str, quantity: float, price: float, total: float) -> None:
        try:
            QUERY  = "INSERT INTO TRADES (TICKER, SIDE, QUANTITY, PRICE, TOTAL, PORTFOLIO) VALUES (%s, %s, %s, %s, %s, %s)"
            VALUES = (ticker, side.lower(), round(quantity, 6), round(price, 4), round(total, 4), round(self.portfolio, 4))
            self.cursor.execute(QUERY, VALUES)
            self.conn.commit()
        except Exception as e:
            self.logging.error(f"LogTrade failed: {e}")
    
    def LivePrice(self, ticker: str) -> float:
        price = rh.stocks.get_latest_price(ticker, priceType=None, includeExtendedHours=True)[0]
        return float(price)
    
    def MarketHours(self) -> bool:
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        current_time = now.hour * 100 + now.minute
        return 930 <= current_time < 1600
    
    def SecondsUntilMarketOpen(self) -> float:
        now  = datetime.now()
        days_ahead = 0
        while True:
            candidate = (now + timedelta(days=days_ahead)).replace(hour=9, minute=30, second=0, microsecond=0)
            if candidate > now and candidate.weekday() < 5:
                return (candidate - now).total_seconds()
            days_ahead += 1
    
    def ClampToMarket(self,dt: datetime) -> float:
            open_dt  = dt.replace(hour=self.market_open[0],  minute=self.market_open[1],  second=0, microsecond=0)
            close_dt = dt.replace(hour=self.market_close[0], minute=self.market_close[1], second=0, microsecond=0)
            if dt <= open_dt:
                return 0.0
            if dt >= close_dt:
                return self.trading_hours
            return (dt - open_dt).total_seconds() / 3600.0
    
    def MarketMinutesElapsed(self, buy_time: datetime, now: datetime) -> float:
        elapsed = 0.0
        current = buy_time.date()
        end = now.date()
        while current <= end:
            if current.weekday() < 5:
                if current == buy_time.date() == now.date():
                    elapsed += self.ClampToMarket(now) - self.ClampToMarket(buy_time)
                elif current == buy_time.date():
                    elapsed += self.trading_hours - self.ClampToMarket(buy_time)
                elif current == end:
                    elapsed += self.ClampToMarket(now)
                else:
                    elapsed += self.trading_hours
            current += timedelta(days=1)
        return max(elapsed, 0.0)
    
    def UpdatePortfolio(self) -> None:
        self.portfolio = self.capital
        prices_raw = rh.stocks.get_latest_price(self.watchlist)
        prices = {t: float(p) for t, p in zip(self.watchlist, prices_raw)}
        for ticker, quantity in self.holdings.items():
            if quantity > 0:
                price = prices.get(ticker)
                if price is None: continue
                self.portfolio += quantity * price
    
    def BuyOrder(self, ticker: str, current_price: float, spend: float, confidence: float = 0.0) -> bool:
        try:
            if (spend > self.capital or self.holdings[ticker] > 0): return False
            quantity  = spend / current_price
            percentage = 1 / (self.holdings[ticker] + quantity)
            # self.purchase_prices[ticker] = (((self.holdings[ticker]/percentage) * self.purchase_prices[ticker]) + ((quantity/percentage) * current_price))
            # self.holdings[ticker] += quantity
            self.purchase_prices[ticker] = current_price
            self.holdings[ticker] = quantity
            self.buy_timestamps[ticker] = datetime.now()
            self.capital -= spend
            self.UpdatePortfolio()
            self.LogTrade(ticker, "buy", quantity, current_price, spend)
            message = (f"BUY {ticker}: {quantity:.4f} shares @ ${current_price:.2f} | Spent ${spend:.2f} | Confidence {confidence:.2%}")
            self.logging.info(message)
            return True
        except Exception as e:
            self.logging.error(f"BuyOrder failed for {ticker}: {e}")
            return False
        
    def SellOrder(self, ticker: str, current_price: float, reason: str = "SIGNAL") -> bool:
        try:
            quantity = self.holdings[ticker]
            if quantity <= 0:
                return False
            proceeds = quantity * current_price
            pnl = proceeds - (quantity * self.purchase_prices[ticker])
            self.holdings[ticker] = 0.0
            self.purchase_prices[ticker] = 0.0
            self.buy_timestamps[ticker] = None
            self.capital += proceeds
            self.UpdatePortfolio()
            self.LogTrade(ticker, "sell", quantity, current_price, proceeds)
            message = (f"SELL {ticker} [{reason}]: {quantity:.4f} shares @ ${current_price:.2f} | Proceeds ${proceeds:.2f} | PnL ${pnl:.2f}")
            self.logging.info(message)
            return True
        except Exception as e:
            self.logging.error(f"SellOrder failed for {ticker}: {e}")
            return False
        
    def Run(self) -> None:
        if not self.Login():
            return
        if not self.SQLConnect():
            self.logging.warning("Running without database.")

        self.logging.info("=" * 50)
        self.logging.info("ALGO TRADER STARTED".center(50))
        self.logging.info("=" * 50)
        try:
            while True:
                if not self.MarketHours():
                    secs = self.SecondsUntilMarketOpen()
                    self.logging.info(f"Market closed — next open in {secs/3600:.1f}h")
                    time.sleep(min(secs, 1800))
                    continue

                now = datetime.now()
                label = f" Scan [{now.strftime('%H:%M:%S')}] "
                self.logging.info(label.center(50, '─'))
                prices_raw = rh.stocks.get_latest_price(self.watchlist)
                prices = {t: float(p) for t, p in zip(self.watchlist, prices_raw)}

                for ticker in self.watchlist:
                    if self.holdings[ticker] > 0:
                        current_price = prices.get(ticker)
                        if current_price is None:
                            continue
                        returns = (current_price - self.purchase_prices[ticker]) / self.purchase_prices[ticker]
                        hours_held = self.MarketMinutesElapsed(self.buy_timestamps[ticker], datetime.now())
                        if hours_held >= self.hold_hours:
                            self.SellOrder(ticker, current_price=current_price, reason="5H_HOLD")
                        elif returns >= self.take_profit:
                            self.SellOrder(ticker, current_price=current_price, reason="TAKE_PROFIT")
                        elif returns <= -self.stop_loss:
                            self.SellOrder(ticker, current_price=current_price, reason="STOP_LOSS")

                open_positions = sum(1 for v in self.holdings.values() if v > 0)
                if open_positions < self.max_positions:
                    predictions = {}
                    for ticker in self.watchlist:
                        if self.holdings[ticker] == 0:
                            features = self.GetLiveFeatures(ticker)
                            if features is None:
                                continue
                            features_scaled = self.scaler.transform(features)
                            predictions[ticker] = float(self.model.predict_probability(features_scaled).flatten()[0])

                    ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                    for ticker, confidence in ranked:
                        if open_positions >= self.max_positions:
                            break
                        if confidence > 0.55:
                            current_price = self.LivePrice(ticker)
                            if self.BuyOrder(ticker, current_price, self.position_size, confidence):
                                open_positions += 1
                                self.no_of_trades.append(self.no_of_trades[-1] + 1)
                            
                self.UpdatePortfolio()
                with open("equitycurve.txt", "a") as f:
                    f.write(f"{self.no_of_trades[-1]},{self.portfolio:.2f}\n")

                pnl = self.portfolio - 100000.0
                pnl_sign = "+" if pnl >= 0 else ""
                self.logging.info(f"Capital: ${self.capital:,.2f}  |  Positions: {open_positions}/{self.max_positions}  |  Portfolio: ${self.portfolio:,.2f}  |  PnL: {pnl_sign}${pnl:,.2f}")
                self.logging.info("─" * 50)
                time.sleep(self.scan_interval)

        except KeyboardInterrupt:
            self.logging.info("Interrupted by user.")

        finally:
            pnl = self.portfolio - 100000.0
            pnl_sign = "+" if pnl >= 0 else ""
            self.logging.info("=" * 50)
            self.logging.info(f"  SESSION COMPLETE")
            self.logging.info(f"  Final Portfolio : ${self.portfolio:,.2f}")
            self.logging.info(f"  Total PnL       : {pnl_sign}${pnl:,.2f}")
            self.logging.info("=" * 50)
            self.Logout()
            self.SQLClose()