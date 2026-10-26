import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.base import clone

from xgboost import XGBClassifier

from src.data.load_data import load_data
from src.artifacts import ROOT, artifact_path
import os
import tempfile
import hashlib
from datetime import datetime, timezone

# Paths
DATA_PATH = ROOT / "data/raw/german.data"
MODEL_DIR = artifact_path().parent
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Business parameters
PROFIT_TN = 15000  # approved good loan; demo utility units, not validated business profit
COST_FN = 10000
COST_FP = 5000

def business_profit(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn * PROFIT_TN - fn * COST_FN - fp * COST_FP

def optimize_threshold(y_true, y_proba):
    best_t = 0
    best_profit = -np.inf

    for t in np.linspace(0.01, 0.99, 100):
        y_pred = (y_proba >= t).astype(int)
        profit = business_profit(y_true, y_pred)

        if profit > best_profit:
            best_profit = profit
            best_t = t

    return best_t

def threshold_stability(X, y, model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    thresholds = []

    for train_idx, val_idx in skf.split(X, y):
        model_clone = clone(model)

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_clone.fit(X_tr, y_tr)
        y_proba = model_clone.predict_proba(X_val)[:, 1]

        t = optimize_threshold(y_val, y_proba)
        thresholds.append(t)

    return np.median(thresholds)

def train():
    print("Loading data...")
    df = load_data(DATA_PATH)
    df['target'] = df['target'].map({1: 0, 2: 1})

    X = df.drop(columns=['target'])
    y = df['target']

    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(
            random_state=42,
            eval_metric='logloss', n_jobs=1
        ))
    ])

    print("Training model...")
    calibrated_model = CalibratedClassifierCV(
        pipeline,
        method='isotonic',
        cv=5
    )

    calibrated_model.fit(X, y)

    print("Optimizing threshold...")
    threshold = threshold_stability(X, y, calibrated_model)

    print(f"Final threshold: {threshold:.3f}")

    print("Saving artifacts...")

    encoders = {col: sorted(X[col].astype(str).unique().tolist()) for col in cat_cols}
    numeric = X.sample(10, random_state=42).copy()
    for col, values in encoders.items():
        numeric[col] = numeric[col].map({v: i for i, v in enumerate(values)})
    bundle = {
        "schema_version": 1, "model": calibrated_model, "threshold": float(threshold),
        "train_scores": calibrated_model.predict_proba(X)[:, 1],
        "feature_names": X.columns.tolist(), "numeric_columns": num_cols,
        "category_encoders": encoders, "background": numeric.to_numpy(dtype=float),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_sha256": hashlib.sha256(Path(DATA_PATH).read_bytes()).hexdigest(),
        "positive_class": "default", "decision_one_means": "reject",
        "fairness_mitigation_applied": False,
    }
    fd, temporary = tempfile.mkstemp(dir=MODEL_DIR, suffix=".joblib")
    os.close(fd)
    try:
        joblib.dump(bundle, temporary)
        os.replace(temporary, artifact_path())
    finally:
        Path(temporary).unlink(missing_ok=True)

    print("Training complete.")

if __name__ == "__main__":
    train()