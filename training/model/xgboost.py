import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
from training.config import SPLIT_CONFIG, PATHS, XGBOOST_CONFIG

def _class_counts(y):
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    return idx_0, idx_1

def _validate_binary_split(y, split_name):
    idx_0, idx_1 = _class_counts(y)
    if len(idx_0) == 0 or len(idx_1) == 0:
        raise ValueError(
            f"{split_name} split must contain both classes. "
            f"Found 0={len(idx_0)}, 1={len(idx_1)}."
        )
    return idx_0, idx_1

def PrepareData(X, Y, future, timestamps):
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=SPLIT_CONFIG["test_size"], shuffle=False
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    joblib.dump(scaler, PATHS["scaler"])

    _validate_binary_split(Y_train, "Training")
    _validate_binary_split(Y_val, "Validation")

    _, future_val     = train_test_split(future,     test_size=SPLIT_CONFIG["test_size"], shuffle=False)
    _, timestamps_val = train_test_split(timestamps, test_size=SPLIT_CONFIG["test_size"], shuffle=False)

    return X_train, Y_train, X_val, Y_val, future_val, timestamps_val

def SaveArtifacts(model, X_val, Y_val, future_val, timestamps_val):
    model.save_model(PATHS["xgboost"])
    np.save(PATHS["X_val"],      X_val)
    np.save(PATHS["Y_val"],      Y_val)
    np.save(PATHS["future"],     np.array(future_val))
    np.save(PATHS["timestamps"], np.array(timestamps_val))

def RunTrainingLoop(X_train, Y_train, X_val, Y_val, future_val, timestamps_val, prompt_retrain=True):
    idx_0, idx_1 = _validate_binary_split(Y_train, "Training")
    scale_pos_weight = len(idx_0) / len(idx_1)
    model = xgb.XGBClassifier(**XGBOOST_CONFIG, scale_pos_weight=scale_pos_weight, use_label_encoder=False)
    model._estimator_type = "classifier"

    while True:
        model.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=False)
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        auc = roc_auc_score(Y_val, y_prob)
        print(f"Validation ROC-AUC: {auc:.3f}")
        print(classification_report(Y_val, y_pred, target_names=["Negative", "Positive"]))

        SaveArtifacts(model, X_val, Y_val, future_val, timestamps_val)

        if not prompt_retrain:
            break

        if input("Retrain? (Y/n): ").lower() != "y":
            break

    return model
