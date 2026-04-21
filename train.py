import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.config import DATA_CONFIG, WATCHLIST
from training.data.fetcher import FetchAll
from training.data.features import BuildDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train the trading model.")
    parser.add_argument(
        "--model-type",
        choices=["neural_network", "xgboost"],
        default="neural_network",
        help="Model pipeline to train.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single training pass without prompting to retrain.",
    )
    return parser.parse_args()


def get_training_pipeline(model_type):
    if model_type == "xgboost":
        from training.model.xgboost import PrepareData, RunTrainingLoop
    else:
        from training.model.trainer import PrepareData, RunTrainingLoop
    return PrepareData, RunTrainingLoop


def main():
    args = parse_args()
    PrepareData, RunTrainingLoop = get_training_pipeline(args.model_type)

    raw_data = FetchAll(WATCHLIST, DATA_CONFIG)
    dataset = BuildDataset(raw_data, WATCHLIST)
    if dataset is None:
        raise SystemExit("No training samples were generated.")

    X, Y, timestamps, future = dataset

    X_tr, Y_tr, X_val, Y_val, fut_val, ts_val = PrepareData(X, Y, future, timestamps)
    RunTrainingLoop(
        X_tr,
        Y_tr,
        X_val,
        Y_val,
        fut_val,
        ts_val,
        prompt_retrain=not args.once,
    )

if __name__ == "__main__":
    main()
