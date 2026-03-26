import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from training.ModularNeuralNetwork import ModularNeuralNet
from training.config import MODEL_CONFIG, SPLIT_CONFIG, PATHS, TRAIN_CONFIG

def PrepareData(X, Y, future, timestamps):
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=SPLIT_CONFIG["test_size"], shuffle=False
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    joblib.dump(scaler, PATHS["scaler"])

    # Balance classes
    idx_0 = np.where(Y_train == 0)[0]
    idx_1 = np.where(Y_train == 1)[0]
    idx_0_sampled = np.random.choice(idx_0, len(idx_1), replace=False)
    balanced = np.random.permutation(np.concatenate([idx_0_sampled, idx_1]))

    _, future_val     = train_test_split(future,     test_size=SPLIT_CONFIG["test_size"], shuffle=False)
    _, timestamps_val = train_test_split(timestamps, test_size=SPLIT_CONFIG["test_size"], shuffle=False)

    return X_train[balanced], Y_train[balanced], X_val, Y_val, future_val, timestamps_val

def SaveArtifacts(model, X_val, Y_val, future_val, timestamps_val):
    model.save_model(PATHS["model"])
    np.save(PATHS["X_val"],     X_val)
    np.save(PATHS["Y_val"],     Y_val)
    np.save(PATHS["future"],    np.array(future_val))
    np.save(PATHS["timestamps"], np.array(timestamps_val))

def RunTrainingLoop(X_train_bal, Y_train_bal, X_val, Y_val, future_val, timestamps_val):
    model = ModularNeuralNet(**MODEL_CONFIG)

    while True:
        model.train(
            X_train_bal, Y_train_bal,
            validation_data=(X_val, Y_val),
            **TRAIN_CONFIG
        )
        metrics, _ = model.evaluate(X_val, Y_val)
        print(metrics)
        SaveArtifacts(model, X_val, Y_val, future_val, timestamps_val)

        if input("Retrain? (Y/n): ").lower() != "y":
            break

    return model