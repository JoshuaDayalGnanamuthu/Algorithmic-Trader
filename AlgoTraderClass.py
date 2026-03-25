from training.ModularNeuralNetwork import ModularNeuralNet
from training.data.features import BuildFeatureVector
from training.utils.math_utils import *
from training.config import WATCHLIST, PATHS
from datetime import datetime, timedelta
import robin_stocks.robinhood as rh
from dataclasses import dataclass
import mysql.connector as my
import numpy as np
import holidays
import logging
import joblib
import time
import os


@dataclass
class TradingConfig:
    hold_hours: float = 5.0
    position_size: float = 5000.0
    max_positions: int = 10
    take_profit: float = 0.03
    stop_loss: float = 0.03
    scan_interval: int = 300
    min_confidence: float = 0.55
    capital: float = 100000.0
    trading_hours: float = 6.5
    market_open: tuple = (9, 30)
    market_close: tuple = (16, 0)
    watchlist: list[str] = WATCHLIST


class PaperTrader:
    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
        self.config = TradingConfig()
        self.portfolio = self.config.capital
        self.holdings = dict.fromkeys(self.config.watchlist, 0.0)
        self.purchase_prices = dict.fromkeys(self.config.watchlist, 0.0)
        self.buy_timestamps = dict.fromkeys(self.config.watchlist, None)
        self.model = ModularNeuralNet.load_model(PATHS["MODEL"])
        self.scaler = joblib.load(PATHS["SCALER"])
        self.conn = None
        self.cursor = None
        self.no_of_trades = [0]
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler("trades.log"), logging.StreamHandler()])
        self.logging = logging.getLogger(__name__)
    
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
    
    def BuyOrder(self, ticker: str, current_price: float, spend: float, confidence: float = 0.0) -> bool:
        try:
            if (spend > self.config.capital or self.holdings[ticker] > 0): return False
            quantity  = spend / current_price
            self.purchase_prices[ticker] = current_price
            self.holdings[ticker] = quantity
            self.buy_timestamps[ticker] = datetime.now()
            self.config.capital -= spend
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
            self.config.capital += proceeds
            self.UpdatePortfolio()
            self.LogTrade(ticker, "sell", quantity, current_price, proceeds)
            message = (f"SELL {ticker} [{reason}]: {quantity:.4f} shares @ ${current_price:.2f} | Proceeds ${proceeds:.2f} | PnL ${pnl:.2f}")
            self.logging.info(message)
            return True
        except Exception as e:
            self.logging.error(f"SellOrder failed for {ticker}: {e}")
            return False
    
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
        ticker_idx = self.config.watchlist.index(ticker) / len(self.config.watchlist)

        return np.array([BuildFeatureVector(closes, highs, lows, volumes, index, ticker_idx)], dtype=float)
    
    def LogTrade(self, ticker: str, side: str, quantity: float, price: float, total: float) -> None:
        if not self.conn or not self.conn.is_connected():
            self.logging.error("DB disconnected, attempting reconnect...")
            if not self.SQLConnect():
                self.logging.error("Failed to reconnect - trade not logged")
                return
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
        if now.date() in holidays.US():
            return False
        return self.config.market_open <= (now.hour, now.minute) < self.config.market_close
    
    def SecondsUntilMarketOpen(self) -> float:
        now  = datetime.now()
        days_ahead = 0
        while True:
            candidate = (now + timedelta(days=days_ahead)).replace(hour=9, minute=30, second=0, microsecond=0)
            if candidate > now and candidate.weekday() < 5:
                return (candidate - now).total_seconds()
            days_ahead += 1
    
    def ClampToMarket(self,dt: datetime) -> float:
            open_dt  = dt.replace(hour=self.config.market_open[0],  minute=self.config.market_open[1],  second=0, microsecond=0)
            close_dt = dt.replace(hour=self.config.market_close[0], minute=self.config.market_close[1], second=0, microsecond=0)
            if dt <= open_dt:
                return 0.0
            if dt >= close_dt:
                return self.config.trading_hours
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
                    elapsed += self.config.trading_hours - self.ClampToMarket(buy_time)
                elif current == end:
                    elapsed += self.ClampToMarket(now)
                else:
                    elapsed += self.config.trading_hours
            current += timedelta(days=1)
        return max(elapsed, 0.0)
    
    def UpdatePortfolio(self) -> None:
        self.portfolio = self.config.capital
        prices_raw = rh.stocks.get_latest_price(self.config.watchlist)
        prices = {t: float(p) for t, p in zip(self.config.watchlist, prices_raw)}
        for ticker, quantity in self.holdings.items():
            if quantity > 0:
                price = prices.get(ticker)
                if price is None: continue
                self.portfolio += quantity * price
    
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
                prices_raw = rh.stocks.get_latest_price(self.config.watchlist)
                prices = {t: float(p) for t, p in zip(self.config.watchlist, prices_raw)}

                for ticker in self.config.watchlist:
                    if self.holdings[ticker] > 0:
                        current_price = prices.get(ticker)
                        if current_price is None:
                            continue
                        returns = (current_price - self.purchase_prices[ticker]) / self.purchase_prices[ticker]
                        hours_held = self.MarketMinutesElapsed(self.buy_timestamps[ticker], datetime.now())
                        if hours_held >= self.config.hold_hours:
                            self.SellOrder(ticker, current_price=current_price, reason="5H_HOLD")
                        elif returns >= self.config.take_profit:
                            self.SellOrder(ticker, current_price=current_price, reason="TAKE_PROFIT")
                        elif returns <= -self.config.stop_loss:
                            self.SellOrder(ticker, current_price=current_price, reason="STOP_LOSS")

                open_positions = sum(1 for v in self.holdings.values() if v > 0)
                if open_positions < self.config.max_positions:
                    predictions = {}
                    for ticker in self.config.watchlist:
                        if self.holdings[ticker] == 0:
                            features = self.GetLiveFeatures(ticker)
                            if features is None:
                                continue
                            features_scaled = self.scaler.transform(features)
                            predictions[ticker] = float(self.model.predict_probability(features_scaled).flatten()[0])

                    ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                    for ticker, confidence in ranked:
                        if open_positions >= self.config.max_positions:
                            break
                        if confidence > 0.55:
                            current_price = self.LivePrice(ticker)
                            if self.BuyOrder(ticker, current_price, self.config.position_size, confidence):
                                open_positions += 1
                                self.no_of_trades.append(self.no_of_trades[-1] + 1)
                            
                self.UpdatePortfolio()
                with open("equitycurve.txt", "a") as f:
                    f.write(f"{self.no_of_trades[-1]},{self.portfolio:.2f}\n")

                pnl = self.portfolio - 100000.0
                pnl_sign = "+" if pnl >= 0 else ""
                self.logging.info(f"Capital: ${self.config.capital:,.2f}  |  Positions: {open_positions}/{self.config.max_positions}  |  Portfolio: ${self.portfolio:,.2f}  |  PnL: {pnl_sign}${pnl:,.2f}")
                self.logging.info("─" * 50)
                time.sleep(self.config.scan_interval)

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