import pandas as pd
from src.monitoring.metrics import compute_basic_metrics, risk_distribution

def test_approval_contract():
    data = pd.DataFrame({"decision": [1, 1, 0], "prediction_proba": [0.9, 0.8, 0.1]})
    assert compute_basic_metrics(data)["approval_rate"] == 0.3333

def test_buckets_include_zero():
    assert sum(risk_distribution(pd.DataFrame({"prediction_proba": [0, 1]})).values()) == 2

def test_empty_metrics_are_finite():
    assert compute_basic_metrics(pd.DataFrame(columns=["decision","prediction_proba"]))["average_risk"] == 0
