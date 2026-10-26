import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

# Payload REAL basado en tu schema
sample_input = {
    "status": "A11",
    "duration": 12,
    "credit_history": "A32",
    "purpose": "A43",
    "credit_amount": 2500,
    "age": 35,
    "savings": "A61",
    "employment_duration": "A72",
    "installment_rate": 4,
    "personal_status_sex": "A93",
    "other_debtors": "A101",
    "present_residence": 4,
    "property": "A121",
    "other_installment_plans": "A143",
    "housing": "A152",
    "number_of_credits": 2,
    "job": "A173",
    "people_liable": 1,
    "telephone": "A191",
    "foreign_worker": "A201"
}


def test_predict():
    response = client.post("/predict", json=sample_input)
    
    assert response.status_code == 200
    
    data = response.json()
    
    assert "default_probability" in data
    assert "prediction" in data
    assert isinstance(data["default_probability"], float)
    assert data["prediction"] in [0, 1]


def test_explain():
    response = client.post("/explain", json=sample_input)
    
    assert response.status_code == 200
    
    data = response.json()
    
    assert "top_positive_factors" in data
    assert "top_negative_factors" in data
    assert isinstance(data["shap_values"], list)
    assert len(data["shap_values"]) == len(sample_input)
    assert abs(data["base_value"] + sum(data["shap_values"]) - data["default_probability"]) < 1e-6

def test_invalid_input_is_rejected():
    assert client.post("/predict", json={**sample_input, "age": -1}).status_code == 422
    assert client.post("/predict", json={**sample_input, "purpose": "unknown"}).status_code == 422


def test_untrained_service_returns_actionable_503(monkeypatch, tmp_path):
    monkeypatch.setenv("CREDIT_MODEL_DIR", str(tmp_path / "untrained"))
    for endpoint in ("/predict", "/explain"):
        response = client.post(endpoint, json=sample_input)
        assert response.status_code == 503
        assert "python -m src.models.train" in response.json()["detail"]


def test_profit_contract_rewards_approved_good_loans():
    from src.models.train import business_profit

    # Two approved good loans, one approved default, one rejected good loan,
    # and one correctly rejected default: 30,000 - 10,000 - 5,000.
    assert business_profit([0, 0, 1, 0, 1], [0, 0, 0, 1, 1]) == 15000
