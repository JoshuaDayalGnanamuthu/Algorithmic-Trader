from AlgoTraderClass import PaperTrader
from dotenv import load_dotenv
import os

load_dotenv("credentials.env")

if __name__ == "__main__":
    trader = PaperTrader(
        username=os.getenv("USERNAME"),
        password=os.getenv("PASSWORD")
    )
    trader.Run()