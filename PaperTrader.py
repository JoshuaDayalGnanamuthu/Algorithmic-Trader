from AlgoTraderClass import PaperTrader
from dotenv import load_dotenv
import os

load_dotenv("credentials.env")

if __name__ == "__main__":
    trader = PaperTrader(
        username=str(os.getenv("USERNAME")),
        password=str(os.getenv("PASSWORD")),
        model_type="xgboost"
    )
    trader.Run()