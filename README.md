# Production-Style Credit Decision System

An end-to-end credit decision system combining calibrated default probabilities, a validation-based decision policy, probability explanations and portfolio monitoring.

## Decision contract

- Target `0`: no default; target `1`: default.
- `prediction=0`: approve; `prediction=1`: reject.
- A case is rejected when its default probability reaches the trained threshold.
- Approval rate counts decisions equal to zero.

The utility matrix rewards an approved good loan, penalizes an approved bad loan, and assigns opportunity cost to rejecting a good loan. Its amounts are explicit assumptions in `src/models/train.py`, not measured lending profit. The threshold is selected using validation folds; it is not a fixed published performance claim.

## Architecture

`German Credit data → preprocessing → calibrated XGBoost → validation-based policy threshold → atomic model bundle → FastAPI → decision log and monitoring`

Training saves a single `models/bundle.joblib` containing the model, threshold, reference scores, explanation encoders/background and data fingerprint. Prediction and drift read that same bundle. It is excluded from Git and regenerated locally or in CI.

## Run

Python 3.12:

```bash
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements.txt
python -m src.models.train
python -m uvicorn src.api.main:app
```

Open `http://127.0.0.1:8000/docs` for the complete input schema. Numeric bounds and categorical codes are validated; unknown fields are rejected. Restart the API after publishing a newly trained bundle.

Optional environment variables: `CREDIT_MODEL_DIR` and `CREDIT_LOG_PATH`. Paths default to locations under the repository root, so execution does not depend on the current working directory.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `/predict` | Calibrated default probability and explicit approve/reject decision |
| `/explain` | SHAP permutation contributions for the same calibrated probability |
| `/metrics` | Decision count, approval rate, mean risk and probability buckets |
| `/drift` | Score PSI against the baseline in the active bundle |
| `/health` | Monitoring state and rule-based alerts |
| `/report` | Business-readable monitoring summary |

An untrained service returns HTTP 503 for prediction/explanation with training instructions. Monitoring reports a no-data state before decisions are recorded.

## Explanations and monitoring

The explanation target is the full calibrated default probability. Numeric category codes are substituted using background observations and decoded before prediction. Explanations preserve source feature names and include their base value and contribution vector. One permutation cycle is an approximate explanation; it is tested for local additivity, not claimed to be causal.

PSI includes tail values outside the baseline range and handles constant reference distributions. A minimum sample requirement prevents interpreting tiny logs as meaningful drift.

The API currently uses a local CSV decision log. A shared deployment requires a transactional log store, authentication and concurrency/load validation.

## Verification

```bash
python -m pytest -q
```

Tests train a temporary bundle and use a temporary log. They cover prediction, input validation, explanation additivity, the approval/rejection contract, probability boundaries and drift edge cases. GitHub Actions repeats these checks with pinned direct dependencies.

## Notebooks and limitations

The notebooks and existing figures preserve earlier experiments. Their metrics and thresholds are historical and do not describe every retraining run. The fairness analysis explores demographic groups; group-specific mitigation is not applied in the API. The historical German Credit sample is a small benchmark and does not establish suitability for contemporary lending decisions.
