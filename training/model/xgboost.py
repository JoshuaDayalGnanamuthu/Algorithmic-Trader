import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
from config import SPLIT_CONFIG, PATHS, TRAIN_CONFIG

def PrepareData(X, Y, future, timestamps):
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=SPLIT_CONFIG["test_size"], shuffle=False
    )

    # XGBoost doesn't need scaling, but we keep it in case other parts of your
    # pipeline depend on the scaler being saved
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    joblib.dump(scaler, PATHS["scaler"])

    # Balance classes by undersampling the majority class
    idx_0 = np.where(Y_train == 0)[0]
    idx_1 = np.where(Y_train == 1)[0]
    idx_0_sampled = np.random.choice(idx_0, len(idx_1), replace=False)
    balanced = np.random.permutation(np.concatenate([idx_0_sampled, idx_1]))

    _, future_val     = train_test_split(future,     test_size=SPLIT_CONFIG["test_size"], shuffle=False)
    _, timestamps_val = train_test_split(timestamps, test_size=SPLIT_CONFIG["test_size"], shuffle=False)

    return X_train[balanced], Y_train[balanced], X_val, Y_val, future_val, timestamps_val

def SaveArtifacts(model, X_val, Y_val, future_val, timestamps_val):
    # Save the XGBoost model using joblib instead of model.save_model()
    model.save_model(PATHS["xgboost"])
    np.save(PATHS["X_val"],      X_val)
    np.save(PATHS["Y_val"],      Y_val)
    np.save(PATHS["future"],     np.array(future_val))
    np.save(PATHS["timestamps"], np.array(timestamps_val))

def RunTrainingLoop(X_train_bal, Y_train_bal, X_val, Y_val, future_val, timestamps_val):
    scale_pos_weight = (Y_train_bal == 0).sum() / (Y_train_bal == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight
    )

    while True:
        model.fit(
            X_train_bal, Y_train_bal,
            eval_set=[(X_val, Y_val)],
            verbose=False,
        )

        # Evaluate on validation set
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        auc = roc_auc_score(Y_val, y_prob)
        print(f"Validation ROC-AUC: {auc:.3f}")
        print(classification_report(Y_val, y_pred, target_names=["Negative", "Positive"]))

        SaveArtifacts(model, X_val, Y_val, future_val, timestamps_val)

        if input("Retrain? (Y/n): ").lower() != "y":
            break

    return model