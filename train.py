import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.config import WATCHLIST, DATA_CONFIG
from training.data.fetcher import FetchAll
from training.data.features import BuildDataset
from training.model.trainer import PrepareData, RunTrainingLoop

if __name__ == "__main__":
    raw_data = FetchAll(WATCHLIST, DATA_CONFIG)
    dataset = BuildDataset(raw_data, WATCHLIST)
    if dataset is None:
        raise SystemExit("No training samples were generated.")

    X, Y, timestamps, future = dataset

    X_tr, Y_tr, X_val, Y_val, fut_val, ts_val = PrepareData(X, Y, future, timestamps)
    RunTrainingLoop(X_tr, Y_tr, X_val, Y_val, fut_val, ts_val)