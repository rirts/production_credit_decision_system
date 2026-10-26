"""Credit decision demo: positive class is default; decision=1 means reject."""
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import shap
from src.api.schemas import CreditInput
from src.artifacts import load_bundle
from src.tracking.logger import log_prediction
from src.monitoring.metrics import load_predictions, compute_basic_metrics, risk_distribution
from src.monitoring.drift import detect_drift
from src.monitoring.alerts import evaluate_alerts
from src.reporting.summary import generate_summary

app = FastAPI()

def bundle_or_503():
    try:
        return load_bundle()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, "Train a compatible bundle with python -m src.models.train") from exc

def frame(data, bundle):
    return pd.DataFrame([data.model_dump()])[bundle["feature_names"]]

@app.get("/")
def root():
    return {"message": "Credit decision demo", "decision_contract": "0=approve, 1=reject"}

@app.post("/predict")
def predict(data: CreditInput):
    b = bundle_or_503()
    proba = float(b["model"].predict_proba(frame(data, b))[0, 1])
    decision = int(proba >= b["threshold"])
    log_prediction(data.model_dump(), proba, decision)
    return {"default_probability": proba, "prediction": decision,
            "decision": "reject" if decision else "approve", "threshold": b["threshold"]}

@app.post("/explain")
def explain(data: CreditInput):
    b = bundle_or_503()
    x = frame(data, b)
    encoded = x.copy()
    for col, categories in b["category_encoders"].items():
        if x.iloc[0][col] not in categories:
            raise HTTPException(422, f"No explanation encoding for category in {col}")
        encoded[col] = encoded[col].map({v: i for i, v in enumerate(categories)})

    def predict_default(values):
        decoded = pd.DataFrame(values, columns=b["feature_names"])
        for col, categories in b["category_encoders"].items():
            decoded[col] = [categories[int(v)] for v in decoded[col]]
        for col in b["numeric_columns"]:
            decoded[col] = pd.to_numeric(decoded[col])
        return b["model"].predict_proba(decoded)[:, 1]

    # Numeric category codes are swapped, not interpolated: explanations target the
    # same calibrated ensemble probability used by /predict, not one base tree.
    explainer = shap.PermutationExplainer(predict_default, b["background"], seed=42)
    result = explainer(encoded.to_numpy(dtype=float), max_evals=2 * len(b["feature_names"]) + 1)
    impacts = np.asarray(result.values[0], dtype=float)
    factors = [{"feature": col, "impact": float(v)} for col, v in zip(b["feature_names"], impacts)]
    return {"explanation_target": "calibrated_default_probability",
            "base_value": float(np.asarray(result.base_values).reshape(-1)[0]),
            "default_probability": float(b["model"].predict_proba(x)[0, 1]),
            "feature_names": b["feature_names"], "shap_values": impacts.tolist(),
            "top_positive_factors": sorted([v for v in factors if v["impact"] > 0], key=lambda v: -v["impact"])[:5],
            "top_negative_factors": sorted([v for v in factors if v["impact"] < 0], key=lambda v: v["impact"])[:5]}

def monitoring_payload():
    try:
        df = load_predictions()
    except FileNotFoundError:
        df = pd.DataFrame(columns=["decision", "prediction_proba"])
    metrics = {**compute_basic_metrics(df), "risk_distribution": risk_distribution(df)}
    try:
        drift = detect_drift()
    except FileNotFoundError:
        drift = {"psi": None, "status": "no_data"}
    alerts = evaluate_alerts(metrics, drift) if metrics["total_predictions"] else []
    return metrics, drift, alerts

@app.get("/metrics")
def get_metrics():
    return monitoring_payload()[0]

@app.get("/drift")
def get_drift():
    return monitoring_payload()[1]

@app.get("/health")
def system_health():
    metrics, drift, alerts = monitoring_payload()
    return {"metrics": metrics, "drift": drift, "alerts": alerts,
            "status": "no_data" if not metrics["total_predictions"] else ("warning" if alerts else "ok")}

@app.get("/report")
def get_report():
    metrics, drift, alerts = monitoring_payload()
    if not metrics["total_predictions"]:
        return {"summary": "No decisions have been recorded.", "status": "no_data"}
    return {**generate_summary(metrics, drift, alerts), "status": "warning" if alerts else "ok"}
