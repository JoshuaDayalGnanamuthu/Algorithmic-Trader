from config import WATCHLIST, DATA_CONFIG
from data.fetcher import FetchAll
from data.features import BuildDataset
from model.trainer import PrepareData, RunTrainingLoop

if __name__ == "__main__":
    raw_data = FetchAll(WATCHLIST, DATA_CONFIG)
    X, Y, timestamps, future = BuildDataset(raw_data, WATCHLIST)

    X_tr, Y_tr, X_val, Y_val, fut_val, ts_val = PrepareData(X, Y, future, timestamps)
    RunTrainingLoop(X_tr, Y_tr, X_val, Y_val, fut_val, ts_val)