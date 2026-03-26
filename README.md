# Algorithmic Trader

A lightweight Python algorithmic trading framework built for backtesting and paper trading with Robinhood, using neural network and XGBoost models.

## Features

- Backtest trading strategy with historical data
- Train models (Modular Neural Network and XGBoost)
- Live paper trading through Robinhood (via robin_stocks)
- Trade logging and MySQL setup script
- Risk controls for stop loss / take profit

## Repository Structure

- `AlgoTraderClass.py` - main trading logic, order execution, features, and portfolio management
- `PaperTrader.py` - entry point for running trading loop
- `training/` - training/backtesting modules and data pipeline
  - `data/` - data fetcher, feature engineering
  - `model/` - NN and XGBoost training scripts
- `setup.py` - initializes MySQL database and tables
- `credentials.env` - environment variables for database and Robinhood login (not tracked)
- `files/` - stores artifacts: scaler, model weights, dataset arrays

## Requirements

- Python 3.10+
- `numpy`, `pandas`, `scikit-learn`, `xgboost`, `joblib`, `python-dotenv`, `mysql-connector-python`, `robin_stocks`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

1. Duplicate `credentials.env.example` (or create `credentials.env`) and add:
   - `USERNAME` / `PASSWORD` (Robinhood credentials)
   - `DATABASE_USERNAME`, `DATABASE_PASSWORD`, `DATABASE_HOST`

2. Verify `training/config.py` watchlist and model paths if needed.

## Setup

Initialize the database once:

```bash
python setup.py
```

## Training

- Neural network training:
  - `python training/model/trainer.py` (or run the training script within `training/`)

- XGBoost training:
  - `python training/model/xgboost.py`

## Backtest

Run backtest logic (adapt path if needed):

```bash
python training/backtest.py
```

## Paper Trading

Run the trading bot with paper trading mode:

```bash
python PaperTrader.py
```

## Notes

- This project is for educational/demo purposes only. Use at your own risk.
- Ensure all required model artifacts (`files/scaler.save`, `files/trader_model.npy`, `files/xgboost_model.json`) exist before live trading.
- Check the logs in `trades.log` and `trader.log` for runtime information.
