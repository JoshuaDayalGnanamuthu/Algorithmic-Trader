"""
Algorithmic Trading Bot using Robinhood API and ML Model
Executes buy/sell orders based on neural network or XGBoost predictions with risk management.
"""

from training.ModularNeuralNetwork import ModularNeuralNet
from training.data.features import BuildFeatureVector
from training.utils.math_utils import *
from training.config import PROJECT_ROOT, WATCHLIST, PATHS
from datetime import datetime, timedelta
import robin_stocks.robinhood as rh
from dataclasses import dataclass, field
import mysql.connector as my
import numpy as np
import xgboost as xgb
import subprocess
import holidays
import logging
import joblib
import time
import os
import sys


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
    auto_retrain: bool = True
    retrain_interval_hours: float = 24.0
    watchlist: list[str] = field(default_factory=lambda: WATCHLIST)


class PaperTrader:
    def __init__(self, username: str, password: str, model_type: str = "neural_network") -> None:
        """
        Initialize PaperTrader with selectable model type.
        
        Args:
            username (str): Robinhood username
            password (str): Robinhood password
            model_type (str): Type of model to use - "neural_network" or "xgboost"
                            Default is "neural_network"
        
        Raises:
            ValueError: If model_type is not "neural_network" or "xgboost"
        """
        if model_type not in ["neural_network", "xgboost"]:
            raise ValueError(f"model_type must be 'neural_network' or 'xgboost', got '{model_type}'")
        
        self.username = username
        self.password = password
        self.model_type = model_type
        self.config = TradingConfig()
        self.initial_capital = self.config.capital
        self.portfolio = self.config.capital
        self.holdings = dict.fromkeys(self.config.watchlist, 0.0)
        self.purchase_prices = dict.fromkeys(self.config.watchlist, 0.0)
        self.buy_timestamps = dict.fromkeys(self.config.watchlist, None)
        self.market_holidays = holidays.US()
        self.conn = None
        self.cursor = None
        self.no_of_trades = [0]
        self.training_process = None
        self.training_log_handle = None
        self.next_retrain_at = None
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler("trades.log", encoding="utf-8"), logging.StreamHandler()])
        self.logging = logging.getLogger(__name__)
        self._load_model()
        self._schedule_initial_retrain()
        self.logging.info(f"Initialized PaperTrader with model_type: {self.model_type}")
    
    def _load_model(self) -> None:
        """Load the selected model type and its scaler."""
        try:
            if self.model_type == "neural_network":
                self.model = ModularNeuralNet.load_model(PATHS["model"])
                self.logging.info("Loaded ModularNeuralNet model")
            elif self.model_type == "xgboost":
                self.model = xgb.XGBClassifier()
                # Recent xgboost builds may omit this until fit/load time.
                self.model._estimator_type = "classifier"
                self.model.load_model(PATHS["xgboost"])
                self.logging.info("Loaded XGBoost model")
            
            self.scaler = joblib.load(PATHS["scaler"])
        except FileNotFoundError as e:
            self.logging.error(f"Failed to load model or scaler: {e}")
            raise
    
    def _predict_probability(self, features_scaled: np.ndarray) -> float:
        """
        Get prediction probability from the selected model.
        
        Args:
            features_scaled (np.ndarray): Scaled feature vector
        
        Returns:
            float: Prediction probability (0-1)
        """
        if self.model_type == "neural_network":
            return float(self.model.predict_probability(features_scaled).flatten()[0])
        elif self.model_type == "xgboost":
            return float(self.model.predict_proba(features_scaled)[0, 1])

    def _model_path_key(self) -> str:
        return "model" if self.model_type == "neural_network" else "xgboost"

    def _training_interval(self) -> timedelta:
        return timedelta(hours=self.config.retrain_interval_hours)

    def _schedule_next_retrain(self, reference_time: datetime | None = None) -> None:
        if not self.config.auto_retrain:
            self.next_retrain_at = None
            return

        base_time = reference_time or datetime.now()
        self.next_retrain_at = base_time + self._training_interval()
        self.logging.info(
            f"Next automatic retraining scheduled for {self.next_retrain_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def _schedule_initial_retrain(self) -> None:
        if not self.config.auto_retrain:
            return

        try:
            last_trained_at = datetime.fromtimestamp(os.path.getmtime(PATHS[self._model_path_key()]))
        except OSError:
            last_trained_at = datetime.now()

        self.next_retrain_at = last_trained_at + self._training_interval()
        self.logging.info(
            f"Automatic retraining enabled; next run at {self.next_retrain_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def _seconds_until_next_retrain(self) -> float | None:
        if not self.config.auto_retrain or self.training_process is not None or self.next_retrain_at is None:
            return None
        return max((self.next_retrain_at - datetime.now()).total_seconds(), 0.0)

    def _training_command(self) -> list[str]:
        return [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--model-type",
            self.model_type,
            "--once",
        ]

    def _maybe_start_retraining(self) -> None:
        if not self.config.auto_retrain or self.training_process is not None or self.next_retrain_at is None:
            return

        if datetime.now() < self.next_retrain_at:
            return

        try:
            log_path = PROJECT_ROOT / "training.log"
            self.training_log_handle = open(log_path, "a", encoding="utf-8")
            self.training_log_handle.write(
                f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting scheduled {self.model_type} retraining.\n"
            )
            self.training_log_handle.flush()
            self.training_process = subprocess.Popen(
                self._training_command(),
                cwd=str(PROJECT_ROOT),
                stdout=self.training_log_handle,
                stderr=subprocess.STDOUT,
            )
            self.logging.info("Started scheduled retraining in the background.")
        except Exception as e:
            self.logging.error(f"Failed to start scheduled retraining: {e}")
            if self.training_log_handle:
                self.training_log_handle.close()
                self.training_log_handle = None
            self.training_process = None
            self._schedule_next_retrain()

    def _poll_retraining_process(self) -> None:
        if self.training_process is None:
            return

        return_code = self.training_process.poll()
        if return_code is None:
            return

        finished_at = datetime.now()
        if self.training_log_handle:
            self.training_log_handle.write(
                f"[{finished_at.strftime('%Y-%m-%d %H:%M:%S')}] Scheduled retraining exited with code {return_code}.\n"
            )
            self.training_log_handle.flush()
            self.training_log_handle.close()
            self.training_log_handle = None

        self.training_process = None

        if return_code == 0:
            try:
                self._load_model()
                self.logging.info("Scheduled retraining completed successfully; model artifacts reloaded.")
            except Exception as e:
                self.logging.error(f"Retraining completed but model reload failed: {e}")
        else:
            self.logging.error(
                f"Scheduled retraining failed with exit code {return_code}. Check training.log for details."
            )

        self._schedule_next_retrain(finished_at)

    def _stop_retraining_process(self) -> None:
        if self.training_process is not None and self.training_process.poll() is None:
            self.logging.info("Stopping scheduled retraining process.")
            self.training_process.terminate()
            try:
                self.training_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.training_process.kill()
                self.training_process.wait(timeout=5)

        self.training_process = None

        if self.training_log_handle:
            self.training_log_handle.close()
            self.training_log_handle = None
    
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

    def SQLConnect(self, load_state: bool = True) -> bool:
        try:
            self.conn = my.connect(host=os.getenv("DATABASE_HOST"), user=os.getenv("DATABASE_USERNAME"),
                            password=os.getenv("DATABASE_PASSWORD"),database="ALGOTRADER")
            if self.conn.is_connected():
                self.logging.info("Conected Successfully to MySQL")
                self.cursor = self.conn.cursor()
                self._ensure_state_tables()
                if load_state and not self.LoadPortfolioState():
                    self.logging.error("Failed to load persisted portfolio state.")
                    self.SQLClose()
                    return False
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

    def _ensure_state_tables(self) -> None:
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS PORTFOLIO_STATE (
            ID TINYINT PRIMARY KEY,
            CAPITAL FLOAT NOT NULL,
            PORTFOLIO FLOAT NOT NULL,
            UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )""")
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS HOLDINGS (
            TICKER VARCHAR(10) PRIMARY KEY,
            QUANTITY FLOAT NOT NULL DEFAULT 0,
            PURCHASE_PRICE FLOAT NOT NULL DEFAULT 0,
            BUY_TIMESTAMP DATETIME NULL,
            UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )""")
        self.conn.commit()

    def _all_state_tickers(self) -> list[str]:
        return list(dict.fromkeys([*self.config.watchlist, *self.holdings.keys()]))

    def _active_tickers(self) -> list[str]:
        active_positions = [ticker for ticker, quantity in self.holdings.items() if quantity > 0]
        return list(dict.fromkeys([*self.config.watchlist, *active_positions]))

    def SavePortfolioState(self) -> bool:
        if not self.conn or not self.conn.is_connected():
            self.logging.error("DB disconnected, attempting reconnect...")
            if not self.SQLConnect(load_state=False):
                self.logging.error("Failed to reconnect - portfolio state not saved")
                return False
        try:
            self.cursor.execute(
                """INSERT INTO PORTFOLIO_STATE (ID, CAPITAL, PORTFOLIO)
                VALUES (1, %s, %s)
                ON DUPLICATE KEY UPDATE
                    CAPITAL = VALUES(CAPITAL),
                    PORTFOLIO = VALUES(PORTFOLIO)""",
                (round(self.config.capital, 4), round(self.portfolio, 4))
            )
            values = [
                (
                    ticker,
                    round(self.holdings.get(ticker, 0.0), 6),
                    round(self.purchase_prices.get(ticker, 0.0), 4),
                    self.buy_timestamps.get(ticker)
                )
                for ticker in self._all_state_tickers()
            ]
            self.cursor.executemany(
                """INSERT INTO HOLDINGS (TICKER, QUANTITY, PURCHASE_PRICE, BUY_TIMESTAMP)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    QUANTITY = VALUES(QUANTITY),
                    PURCHASE_PRICE = VALUES(PURCHASE_PRICE),
                    BUY_TIMESTAMP = VALUES(BUY_TIMESTAMP)""",
                values
            )
            self.conn.commit()
            return True
        except Exception as e:
            self.logging.error(f"SavePortfolioState failed: {e}")
            return False

    def LoadPortfolioState(self) -> bool:
        if not self.conn or not self.conn.is_connected():
            return False
        try:
            self.cursor.execute("SELECT CAPITAL, PORTFOLIO FROM PORTFOLIO_STATE WHERE ID = 1")
            account_row = self.cursor.fetchone()
            self.cursor.execute("SELECT TICKER, QUANTITY, PURCHASE_PRICE, BUY_TIMESTAMP FROM HOLDINGS")
            holdings_rows = self.cursor.fetchall()

            for ticker in list(self.holdings.keys()):
                self.holdings[ticker] = 0.0
                self.purchase_prices[ticker] = 0.0
                self.buy_timestamps[ticker] = None

            open_positions = 0
            for ticker, quantity, purchase_price, buy_timestamp in holdings_rows:
                if ticker not in self.holdings:
                    self.holdings[ticker] = 0.0
                    self.purchase_prices[ticker] = 0.0
                    self.buy_timestamps[ticker] = None
                self.holdings[ticker] = float(quantity)
                self.purchase_prices[ticker] = float(purchase_price)
                self.buy_timestamps[ticker] = buy_timestamp
                if quantity > 0:
                    open_positions += 1

            if account_row:
                self.config.capital = float(account_row[0])
                self.portfolio = float(account_row[1])
            else:
                self.portfolio = self.config.capital
                self.SavePortfolioState()
                self.logging.info("Initialized portfolio state in database.")
                return True

            if open_positions > 0:
                try:
                    self.UpdatePortfolio()
                    self.SavePortfolioState()
                except Exception as e:
                    self.logging.warning(f"Loaded holdings from database but could not refresh portfolio value: {e}")
            else:
                self.portfolio = self.config.capital

            self.logging.info(
                f"Loaded portfolio state from database: ${self.portfolio:,.2f} across {open_positions} open positions."
            )
            return True
        except Exception as e:
            self.logging.error(f"LoadPortfolioState failed: {e}")
            return False
    
    def BuyOrder(self, ticker: str, current_price: float, spend: float, confidence: float = 0.0) -> bool:
        try:
            if (spend > self.config.capital or self.holdings[ticker] > 0): return False
            quantity  = spend / current_price
            self.purchase_prices[ticker] = current_price
            self.holdings[ticker] = quantity
            self.buy_timestamps[ticker] = datetime.now()
            self.config.capital -= spend
            self.UpdatePortfolio()
            self.SavePortfolioState()
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
            self.SavePortfolioState()
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
            if not self.SQLConnect(load_state=False):
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

    def IsTradingDay(self, day) -> bool:
        return day.weekday() < 5 and day not in self.market_holidays
    
    def MarketHours(self) -> bool:
        now = datetime.now()
        if not self.IsTradingDay(now.date()):
            return False
        return self.config.market_open <= (now.hour, now.minute) < self.config.market_close
    
    def SecondsUntilMarketOpen(self) -> float:
        now  = datetime.now()
        days_ahead = 0
        while True:
            candidate = (now + timedelta(days=days_ahead)).replace(hour=9, minute=30, second=0, microsecond=0)
            if candidate > now and self.IsTradingDay(candidate.date()):
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
            if self.IsTradingDay(current):
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
        tracked_tickers = self._active_tickers()
        if not tracked_tickers:
            return
        prices_raw = rh.stocks.get_latest_price(tracked_tickers)
        prices = {
            ticker: float(price)
            for ticker, price in zip(tracked_tickers, prices_raw)
            if price is not None
        }
        for ticker, quantity in self.holdings.items():
            if quantity > 0:
                price = prices.get(ticker)
                if price is None: continue
                self.portfolio += quantity * price
    
    def Run(self) -> None:
        if not self.Login():
            return
        if not self.SQLConnect():
            self.logging.error("Database connection and portfolio state load are required before trading can start.")
            self.Logout()
            return

        self.logging.info("=" * 50)
        self.logging.info("ALGO TRADER STARTED".center(50))
        self.logging.info(f"Model Type: {self.model_type.upper()}".center(50))
        self.logging.info("=" * 50)
        try:
            while True:
                self._poll_retraining_process()
                self._maybe_start_retraining()

                if not self.MarketHours():
                    secs = self.SecondsUntilMarketOpen()
                    self.logging.info(f"Market closed - next open in {secs/3600:.1f}h")
                    sleep_for = min(secs, 1800)
                    retrain_wait = self._seconds_until_next_retrain()
                    if retrain_wait is not None:
                        sleep_for = min(sleep_for, retrain_wait)
                    time.sleep(max(sleep_for, 1))
                    continue

                now = datetime.now()
                label = f" Scan [{now.strftime('%H:%M:%S')}] "
                self.logging.info(label.center(50, '-'))
                tracked_tickers = self._active_tickers()
                prices_raw = rh.stocks.get_latest_price(tracked_tickers)
                prices = {
                    ticker: float(price)
                    for ticker, price in zip(tracked_tickers, prices_raw)
                    if price is not None
                }

                for ticker, quantity in list(self.holdings.items()):
                    if quantity > 0:
                        current_price = prices.get(ticker)
                        if current_price is None:
                            continue
                        returns = (current_price - self.purchase_prices[ticker]) / self.purchase_prices[ticker]
                        buy_time = self.buy_timestamps.get(ticker)
                        if buy_time is None:
                            continue
                        hours_held = self.MarketMinutesElapsed(buy_time, datetime.now())
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
                            predictions[ticker] = self._predict_probability(features_scaled)

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
                self.SavePortfolioState()
                with open("equitycurve.txt", "a") as f:
                    f.write(f"{self.no_of_trades[-1]},{self.portfolio:.2f}\n")

                pnl = self.portfolio - self.initial_capital
                pnl_sign = "+" if pnl >= 0 else ""
                self.logging.info(f"Capital: ${self.config.capital:,.2f}  |  Positions: {open_positions}/{self.config.max_positions}  |  Portfolio: ${self.portfolio:,.2f}  |  PnL: {pnl_sign}${pnl:,.2f}")
                self.logging.info("-" * 50)
                sleep_for = self.config.scan_interval
                retrain_wait = self._seconds_until_next_retrain()
                if retrain_wait is not None:
                    sleep_for = min(sleep_for, retrain_wait)
                time.sleep(max(sleep_for, 1))

        except KeyboardInterrupt:
            self.logging.info("Interrupted by user.")

        finally:
            self._stop_retraining_process()
            self.UpdatePortfolio()
            self.SavePortfolioState()
            pnl = self.portfolio - self.initial_capital
            pnl_sign = "+" if pnl >= 0 else ""
            self.logging.info("=" * 50)
            self.logging.info(f"  SESSION COMPLETE")
            self.logging.info(f"  Final Portfolio : ${self.portfolio:,.2f}")
            self.logging.info(f"  Total PnL       : {pnl_sign}${pnl:,.2f}")
            self.logging.info("=" * 50)
            self.Logout()
            self.SQLClose()
