# MP270 – Hostel Mess Food Waste Prediction

## 1. Project Overview

**Group ID:** G029  
**Problem ID:** MP270  
**Project:** Hostel Mess Food Waste Prediction  
**Location:** Surat, Gujarat, India  
**Course:** SCSE3040 – MLOps Mini Project  
**Faculty:** Amit Soni  
**Batch:** 69

### Team Members

| Name | Enrollment Number |
|---|---|
| Shreyanshi Sharma | S24CSEU2596 |
| Mayank Bijalwan | S24CSEU0269 |
| Vanshika Gupta | S24CSEU2597 |
| Sanskar Srivastava | S24CSEU2577 |

The objective of this project is to predict hostel mess food waste (waste_kg) using operational and meal-related features. The solution follows an MLOps workflow covering data quality assessment, preprocessing, model training, experiment tracking, model evaluation, retraining, API development, containerization, and deployment.

## 2. Problem Statement

Hostel messes often prepare more food than students consume, resulting in food waste. This project uses machine learning to estimate food waste based on:

- Number of registered students
- Attendance percentage
- Menu rating
- Temperature
- Cooked food quantity
- Plate size
- Meal type
- Hostel

The predictions can help mess administrators make better food preparation decisions.

## 3. Project Structure
```text
G029-MP270-Hostel-Food-Waste-MLOps/
│
├── .dvc/
│   └── config
├── data/
│   ├── raw/
│   │   ├── train.csv.dvc
│   │   ├── later.csv.dvc
│   │   └── holdout_features.csv
│   └── processed/
│
├── models/
│   ├── model.joblib
│   └── v1_model.joblib
│
├── reports/
│   ├── data_quality_audit.csv
│   ├── model_comparison.csv
│   ├── psi_report.csv
│   ├── mae_by_hostel.csv
│   ├── v1_v2_comparison.csv
│   ├── monitoring_events.jsonl
│   └── findings.yaml
│
├── predictions/
│   └── holdout_predictions.csv
│
├── src/
│   └── mp270/
│       ├── preprocessing.py
│       ├── train.py
│       ├── evaluate.py
│       ├── retrain_v2.py
│       ├── predict_holdout.py
│       └── api.py
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_api.py
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── params.yaml
├── requirements.txt
├── mlflow.db
└── README.md
```

## 4. Data Quality and Preprocessing

The project includes data quality checks and preprocessing operations.

Key findings:

- 59 duplicate rows removed.
- 11 target-value errors corrected by dividing the affected `waste_kg` values by 10.
- Meal spelling variants normalized.
- No identified sentinel values, unit mix-ups, or stuck sensor values.
- `waste_weighed_at_bin` excluded because it is a post-meal measurement and causes data leakage.

Feature preprocessing includes:
- Date feature extraction.
- Missing-value imputation.
- Numerical feature scaling.
- Categorical feature encoding.

## 5. Model Training

Three regression algorithms were evaluated:

1. Decision Tree Regressor
2. Random Forest Regressor
3. Gradient Boosting Regressor

The best-performing initial model was Gradient Boosting.

Initial validation results:

| Model | MAE |
|---|---:|
| Decision Tree | 6.1433 |
| Random Forest | 3.9581 |
| Gradient Boosting | 3.4931 |

Model parameters are maintained in `params.yaml`.

## 6. Experiment Tracking and Model Registry

MLflow is used for experiment tracking and model registration.

- V1 experiment: `MP270-Hostel-Food-Waste`
- V2 experiment: `MP270-Hostel-Food-Waste-Person3`
- Champion run ID: `76bcc42f53d54f9e91c0fbf9b93647eb`

Tracked information includes:
- Model parameters
- MAE, RMSE, and R² metrics
- Training data SHA-256 fingerprint
- Registered model versions

Registered model:

`MP270-Hostel-Food-Waste-Model`

The `champion` alias identifies the selected production model.

The champion model was promoted to version 5 following retraining evaluation.

## 7. Model Monitoring and Retraining

The project evaluates data drift using Population Stability Index (PSI).

Model performance is also evaluated separately for each hostel.

- Most drifted feature: `cooked_kg` (PSI: 3.5053).
- Worst-served group: Boys Hostel 2 (MAE: 24.6421).

Version 2 retraining uses:
- The cleaned original training dataset.
- The first 70% of the later dataset for retraining.
- The final 30% for evaluation.

Results:

| Metric | Result |
|---|---:|
| V1 MAE | 13.4519 |
| V2 MAE | 6.0482 |
| Improvement | 55.04% |

The retrained model achieved more than the required 5% MAE improvement and was promoted as the champion.

## 8. Holdout Predictions

The champion model generates predictions for the holdout dataset.

Output file:

`predictions/holdout_predictions.csv`

Required output columns:

- `record_id`
- `prediction`

The generated holdout output contains 250 predictions.

## 9. API Development

The project uses FastAPI to expose the trained model.

### Available endpoints

**Health check**

`GET /health`

**Prediction**

`POST /predict`

The prediction endpoint accepts:

```json
{
  "students_registered": 500,
  "attendance_pct": 85,
  "menu_rating": 4,
  "temp_c": 30,
  "cooked_kg": 400,
  "plate_size_cm": 25,
  "meal": "lunch",
  "hostel": "Girls Hostel"
}
```

Validation rules:
- `cooked_kg` must not exceed 1000.
- `plate_size_cm` must be at least 10.
- Invalid requests return HTTP 422.

The API uses 30 March 2026 as a fixed reference date because the client request contract does not include a date.

### JSON Logging

Each successful prediction generates one structured JSON log on stdout using Python's logging module and a JSON formatter. Customer details are not logged.

### Run the API

```bash
uvicorn src.mp270.api:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

`http://127.0.0.1:8000/docs`

## 10. Testing and Continuous Integration

Automated tests are implemented using pytest and FastAPI TestClient.

The test suite covers:
- Data preprocessing.
- Duplicate removal.
- Date feature generation.
- API health endpoint.
- Prediction endpoint.
- Input validation.

Current test result: **7 tests passed.**

GitHub Actions runs the automated test suite on every push and pull request.

## 11. Docker and Kubernetes Deployment

The deployment configuration targets:

- Docker image: `mp270-api:1.0`
- Kubernetes Deployment: `mp270-api`
- Kubernetes Service: `mp270-svc`
- Replicas: 3
- Container port: 8000
- Service port: 80
- Health probes: `/health`

The API is designed to run as a containerized service and can be deployed to a Kubernetes cluster.

Final Docker and Kubernetes deployment tests passed, including all three replicas, health checks, predictions and JSON logging.

## 12. Installation and Setup

Clone the repository:

```bash
git clone https://github.com/vanshiiikagupta/G029-MP270-Hostel-Food-Waste-MLOps.git
cd G029-MP270-Hostel-Food-Waste-MLOps
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run automated tests:

```bash
python -m pytest -q
```

## 13. MLOps Tools and Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- MLflow
- DVC
- FastAPI
- Docker
- Kubernetes
- Git and GitHub
- GitHub Actions
- Pytest

## 14. MLflow Data Provenance

The three recent V1 training runs contain verified
SHA-256 fingerprints of the original training dataset.
Four earlier experimental runs lack data fingerprints
because their exact source dataset could not be verified.
These runs have been preserved without modification
to maintain the integrity of the experiment history.

The earlier V2 registration run lacks complete tracking metadata. The current champion registration contains verified dataset fingerprints and evaluation metrics.

MLflow artifacts are stored locally. The committed Joblib models support inference independently of those artifact locations. DVC has no configured remote, so reproducing training from a fresh clone requires the original instructor-provided datasets or an existing DVC cache.

## 15. Conclusion

This project implements an end-to-end MLOps workflow for hostel mess food waste prediction. It includes data quality management, reproducible preprocessing, model experimentation, drift monitoring, retraining, model versioning, API development, automated testing, and deployment configuration.

The retrained champion model achieved a 55.04% reduction in MAE compared with the V1 model on the specified V2 evaluation split.
