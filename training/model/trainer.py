import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from training.ModularNeuralNetwork import ModularNeuralNet
from training.config import MODEL_CONFIG, SPLIT_CONFIG, PATHS, TRAIN_CONFIG

def _class_counts(y):
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    return idx_0, idx_1

def _class_weights(y, split_name):
    idx_0, idx_1 = _class_counts(y)
    if len(idx_0) == 0 or len(idx_1) == 0:
        raise ValueError(
            f"{split_name} split must contain both classes. "
            f"Found 0={len(idx_0)}, 1={len(idx_1)}."
        )

    total_count = len(y)
    return {
        0: total_count / (2 * len(idx_0)),
        1: total_count / (2 * len(idx_1)),
    }

def PrepareData(X, Y, future, timestamps):
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=SPLIT_CONFIG["test_size"], shuffle=False
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    joblib.dump(scaler, PATHS["scaler"])
    _class_weights(Y_train, "Training")

    _, future_val     = train_test_split(future,     test_size=SPLIT_CONFIG["test_size"], shuffle=False)
    _, timestamps_val = train_test_split(timestamps, test_size=SPLIT_CONFIG["test_size"], shuffle=False)

    return X_train, Y_train, X_val, Y_val, future_val, timestamps_val

def SaveArtifacts(model, X_val, Y_val, future_val, timestamps_val):
    model.save_model(PATHS["model"])
    np.save(PATHS["X_val"],     X_val)
    np.save(PATHS["Y_val"],     Y_val)
    np.save(PATHS["future"],    np.array(future_val))
    np.save(PATHS["timestamps"], np.array(timestamps_val))

def RunTrainingLoop(X_train, Y_train, X_val, Y_val, future_val, timestamps_val):
    model = ModularNeuralNet(**MODEL_CONFIG)
    class_weights = _class_weights(Y_train, "Training")

    while True:
        model.train(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            class_weights=class_weights,
            **TRAIN_CONFIG
        )
        metrics, _ = model.evaluate(X_val, Y_val)
        print(metrics)
        SaveArtifacts(model, X_val, Y_val, future_val, timestamps_val)

        if input("Retrain? (Y/n): ").lower() != "y":
            break

    return model
