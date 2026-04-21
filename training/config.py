from pathlib import Path


WATCHLIST = [
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
    "SPY", 
    "QQQ",
    "SOXX",   # Semiconductor ETF
    "ARKK",   # ARK Innovation ETF
    "IWM",    # Russell 2000
    "XLK",    # Tech Sector ETF
]

DATA_CONFIG = {
    "interval": "hour",
    "span": "3month",
    "bounds": "regular",
    "min_bars": 60,
    "warmup": 50,
    "forward_bars": 5,
}

LABEL_CONFIG = {
    "buy_threshold":  0.015,
    "sell_threshold": -0.015,
}

MODEL_CONFIG = {
    "input_size": 18,
    "hidden_layers": [64, 32, 16],
    "activation": "relu",
    "final_activation": "sigmoid",
}

TRAIN_CONFIG = {
    "epochs": 5000,
    "learning_rate": 0.0005,
    "batch_size": 64,
    "learning_rate_decay": 0.999,
    "decay_interval": 50,
    "early_stopping_patience": 150,
    "print_interval": 100,

}

XGBOOST_CONFIG = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "eval_metric": "auc",
    "subsample": 0.5,
    "colsample_bytree": 0.5,
    "colsample_bylevel": 0.5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0
}

SPLIT_CONFIG = {
    "test_size": 0.2,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILES_DIR = PROJECT_ROOT / "files"

PATHS = {
    "scaler": str(FILES_DIR / "scaler.save"),
    "model": str(FILES_DIR / "trader_model.npy"),
    "X_val": str(FILES_DIR / "X_validate.npy"),
    "Y_val": str(FILES_DIR / "Y_validate.npy"),
    "future": str(FILES_DIR / "future_returns.npy"),
    "timestamps": str(FILES_DIR / "timestamps_val.npy"),
    "xgboost": str(FILES_DIR / "xgboost_model.json"),
}
