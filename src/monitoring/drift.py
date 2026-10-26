import pandas as pd
import numpy as np
from pathlib import Path
import joblib


def load_train_scores():
    from src.artifacts import load_bundle
    return load_bundle()["train_scores"]


def load_production_scores():
    from src.monitoring.metrics import LOG_PATH
    path = LOG_PATH
    
    if not path.exists():
        raise FileNotFoundError("No production data found")
    
    df = pd.read_csv(path)
    return df["prediction_proba"].values

def calculate_psi(expected, actual, bins=10):
    expected = np.array(expected)
    actual = np.array(actual)
    if not len(expected) or not len(actual) or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("PSI requires non-empty finite scores")

    breakpoints = np.linspace(0, 100, bins + 1)
    breakpoints = np.unique(np.percentile(expected, breakpoints))
    if len(breakpoints) < 2:
        breakpoints = np.array([-np.inf, float(expected[0]), np.inf])
    else:
        breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    # evitar division por cero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))

    return psi

def detect_score_drift(train_scores, production_scores):
    psi = calculate_psi(train_scores, production_scores)

    if psi < 0.1:
        status = "stable"
    elif psi < 0.25:
        status = "moderate_drift"
    else:
        status = "significant_drift"

    return {
        "psi": round(float(psi), 4),
        "status": status
    }

def detect_drift(min_samples=50):
    train_scores = load_train_scores()
    prod_scores = load_production_scores()

    if len(prod_scores) < min_samples:
        return {
            "psi": None,
            "status": "insufficient_data",
            "message": f"Need at least {min_samples} samples, got {len(prod_scores)}"
        }

    result = detect_score_drift(train_scores, prod_scores)

    result["train_mean"] = round(float(train_scores.mean()), 4)
    result["production_mean"] = round(float(prod_scores.mean()), 4)
    result["sample_size"] = len(prod_scores)

    return result