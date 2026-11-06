# MLOps Lab: Modular ML Pipelines, FastAPI & Docker (Sessions 7, 8 & 9)

**Student Name:** Arjit Prasad Das  
**Roll Number:** 24AI013  
**Course:** B.Tech, Artificial Intelligence & Data Science  
**Subject:** MLOps: From Machine Learning to Production AI Systems  
**GitHub Repository:** [https://github.com/arjit-prasad-das/ML.OPS_last](https://github.com/arjit-prasad-das/ML.OPS_last)  

---

## 1. Project Overview & Objectives

This project refactors a machine learning project for **House Price Prediction** into a modular, testable, and reproducible software application. It covers:
1. **Modular ML Pipeline (Session 7):** Decoupling ingestion, cleaning, training, evaluation, and inference, with preprocessing encapsulated inside a Scikit-learn Pipeline to guarantee identical transformations without data leakage.
2. **Model Serving with FastAPI (Session 8):** Exposing low-latency REST endpoints (`GET /health`, `POST /predict`) with strict Pydantic input validation, boundary checking, and OpenAPI Swagger documentation.
3. **Containerization with Docker (Session 9):** Packaging the trained model and inference service into a portable Docker container with verified port forwarding, container logging, and host-to-container prediction parity.
4. **Multi-Service Architecture Design:** Architecting a multi-service Docker Compose specification uniting FastAPI with an MLflow Tracking Server and persistent storage.

---

## 2. Technology Stack

* **Language:** Python 3.11
* **Machine Learning:** Scikit-learn (`StandardScaler`, `RandomForestRegressor`, `Pipeline`), Joblib
* **Data Processing:** Pandas, NumPy
* **API & Serving:** FastAPI, Pydantic, Uvicorn, HTTPX
* **Testing & Quality:** Pytest
* **Containerization:** Docker, Docker Compose
* **Experimentation & Tracking:** MLflow
* **Visualization & Reporting:** Matplotlib, ReportLab

---

## 3. Project Architecture

```
Raw CSV Dataset (data/raw/dataset.csv)
       │
       ▼
Data Cleaning & Non-Leaking Train/Test Split (src/data.py)
       │
       ▼
Unified Preprocessing + Model Pipeline Training (src/train.py)
   [StandardScaler -> RandomForestRegressor]
       │
       ▼
Export Serialized Pipeline Artifact (models/house_price_pipeline.joblib)
       │
       ├──────────────────────────────┬─────────────────────────────┐
       ▼                              ▼                             ▼
Held-Out Evaluation           Batch / Record Scoring         Containerized REST API
 (src/evaluate.py)               (src/predict.py)                (src/api.py)
  • MAE, RMSE, R²                 • 5 Sample Records             • GET /health
  • Error Residual Plots          • sample_predictions.json      • POST /predict
                                                                 • Pydantic Schemas
                                                                    │
                                                                    ▼
                                                            Docker Container
                                                          (ml-inference-api:1.0)
```

---

## 4. Repository Structure

```text
mlops-lab-7-9/
├── configs/
│   └── config.yaml              # Centralized YAML configuration for dataset, hyperparameters & paths
├── data/
│   ├── raw/
│   │   └── dataset.csv          # Versioned raw house price dataset (240 records, 5 features + target)
│   └── processed/
│       ├── train.csv            # Cleaned training split (192 records, 80%)
│       └── test.csv             # Cleaned held-out test split (49 records, 20%)
├── models/
│   └── house_price_pipeline.joblib # Serialized end-to-end Scikit-learn inference pipeline
├── reports/
│   ├── metrics.json             # Held-out evaluation metrics (MAE, RMSE, R²)
│   ├── sample_predictions.json  # Predictions generated for 5 sample records
│   ├── prediction_comparison.json # Local vs Docker container prediction equivalence table
│   ├── generate_report_pdf.py   # Script generating Report.pdf
│   └── figures/
│       └── actual_vs_predicted.png # Regression scatter plot
├── screenshots/
│   ├── 01_pipeline_training_terminal.png # Training & evaluation terminal execution
│   ├── 02_swagger_ui_docs.png           # Interactive Swagger UI browser screenshot
│   ├── 03_health_check_response.png     # Health endpoint browser screenshot
│   ├── 05_docker_run_and_logs.png       # Docker build, run, and logs screenshot
│   ├── 06_curl_and_validation_terminal.png # Validation & error handling testing
│   ├── 07_prediction_parity_terminal.png # Local vs Docker parity proof
│   └── 08_pytest_passing_terminal.png   # Full 9/9 pytest suite passing
├── src/
│   ├── __init__.py
│   ├── data.py                  # Loading, validation, cleaning, and train/test splitting
│   ├── train.py                 # Constructs unified Scikit-learn Pipeline and fits on train split
│   ├── evaluate.py              # Evaluates model on held-out test data and exports reports
│   ├── predict.py               # Loads pipeline and scores single/batch input dictionaries
│   ├── api.py                   # Production FastAPI REST application with Pydantic validation
│   └── verify_predictions.py    # Local vs Docker verification test harness
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py         # Automated pipeline tests (schema, leakage, shapes, validation)
│   └── test_api.py              # Automated API tests (health, 200 predict, 422 errors)
├── Dockerfile                   # Multi-stage optimized Docker build definition
├── .dockerignore                # Excludes local caches, virtual environments, and raw data
├── docker-compose.yml           # Multi-service composition for FastAPI & MLflow Tracking Server
├── requirements.txt             # Exact locked dependencies
├── Report.pdf                   # Formal academic submission report
├── .gitignore                   # Version control exclusions
└── README.md                    # Complete reproduction guide
```

---

## 5. Dataset Description

* **Domain:** Residential Real Estate Valuation (Regression).
* **Source:** Deterministic synthetic dataset (240 observations, 6 columns). Contains no personal or confidential information.
* **Target Variable:** `price` (Continuous float value in USD).
* **Input Features:**
  1. `area_sqft` (float): Total floor space in square feet (550 to 3,200).
  2. `bedrooms` (integer): Number of bedrooms (1 to 6).
  3. `bathrooms` (integer): Number of bathrooms (1 to 4).
  4. `age_years` (integer): Age of property in years (0 to 60).
  5. `distance_km` (float): Distance to metropolitan center in kilometers (0.5 to 25.0).

---

## 6. Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/arjit-prasad-das/ML.OPS_last.git
cd ML.OPS_last

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 7. Execution Guide: Part A — ML Pipelines

### 1. Train the Pipeline
```bash
python -m src.train
```
Loads raw data, performs schema validation, splits dataset into 192 training rows and 49 testing rows without data leakage, fits the `Pipeline([('scaler', StandardScaler()), ('regressor', RandomForestRegressor())])`, and persists the artifact to `models/house_price_pipeline.joblib`.

### 2. Evaluate Held-Out Performance
```bash
python -m src.evaluate
```
Scores `data/processed/test.csv`, exports metrics to `reports/metrics.json`, and saves `reports/figures/actual_vs_predicted.png`.

**Achieved Evaluation Metrics:**
* **$R^2$ Score:** `0.9837` *(Explains 98.37% of price variance)*
* **MAE:** `$17,227.05`
* **RMSE:** `$21,353.52`

### 3. Generate Sample Predictions
```bash
python -m src.predict
```
Scores 5 distinct property records and records output in `reports/sample_predictions.json`.

### 4. Run Pipeline Unit Tests
```bash
pytest tests/test_pipeline.py -v
```

---

## 8. Execution Guide: Part B — FastAPI REST Service

### 1. Launch FastAPI Server Locally
```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```
* **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **OpenAPI Schema:** [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

### 2. Test Endpoints

#### A. Health Check (`GET /health`)
```bash
curl -X GET http://127.0.0.1:8000/health
```
```json
{
  "status": "ok",
  "service": "House Price Prediction API",
  "version": "1.0.0",
  "model_loaded": true
}
```

#### B. Prediction Endpoint (`POST /predict`)
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area_sqft": 1800.0,
    "bedrooms": 3,
    "bathrooms": 2,
    "age_years": 5,
    "distance_km": 12.0
  }'
```
```json
{
  "predicted_price": 423884.13,
  "currency": "USD",
  "status": "success"
}
```

#### C. Validation & Error Handling (HTTP 422)
* **Missing Field:** Payload omitting `distance_km` returns HTTP 422 (`"Field required"`).
* **Invalid Data Type:** Payload with `"area_sqft": "one thousand"` returns HTTP 422 (`"unable to parse string as a number"`).
* **Business Constraint Violation:** Payload with `"area_sqft": -500.0` or `"bedrooms": 0` returns HTTP 422 (`"Input should be greater than 0"`).

### 3. Run Automated API Tests
```bash
pytest tests/test_api.py -v
```

---

## 9. Execution Guide: Part C — Dockerization & Compose

### 1. Build the Docker Image
```bash
docker build -t ml-inference-api:1.0 .
```

### 2. Verify Docker Image
```bash
docker image ls ml-inference-api:1.0
```

### 3. Run the Container
```bash
docker run -d --name ml-api -p 8000:8000 ml-inference-api:1.0
```

### 4. Inspect Container Status & Logs
```bash
# Check container is up
docker ps

# Stream logs
docker logs ml-api
```

### 5. Stop and Clean Up Container
```bash
docker stop ml-api
docker rm ml-api
```

### 6. Multi-Service Docker Compose Status (Section C6)
In accordance with Page 6 of the assignment specification:
* **Implemented & Verified Work:** Standalone containerized FastAPI service (`ml-inference-api:1.0`) with model loading, port forwarding, and verified prediction parity.
* **Researched & Designed Architecture (`docker-compose.yml`):** Multi-service stack connecting `api` (FastAPI) to `mlflow` (`ghcr.io/mlflow/mlflow:v2.11.3`) tracking server over an internal bridge network with persistent SQLite and artifact storage (`mlflow-data`). Validated with `docker compose config`.

```bash
docker compose config
```

---

## 10. Part D — Local vs. Docker Numerical Parity

To verify deployment reproducibility:
```bash
python -m src.verify_predictions
```
All 5 sample test cases demonstrate exact numerical parity between the local Python environment and the Docker REST API:
$$\Delta = |y_{\text{local}} - y_{\text{docker}}| = \$0.0000$$

| Case ID | Input Features | Local Pipeline ($) | Docker API ($) | Absolute Difference | Parity Status |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | 1050 sqft, 2 beds, 1 bath, 10 yrs, 5.2 km | $251,052.91 | $251,052.91 | $0.0000 | **EXACT MATCH** |
| **2** | 1800 sqft, 3 beds, 2 baths, 5 yrs, 12.0 km | $423,884.13 | $423,884.13 | $0.0000 | **EXACT MATCH** |
| **3** | 2400 sqft, 4 beds, 3 baths, 15 yrs, 3.5 km | $553,672.70 | $553,672.70 | $0.0000 | **EXACT MATCH** |
| **4** | 850 sqft, 1 bed, 1 bath, 25 yrs, 18.4 km | $217,335.68 | $217,335.68 | $0.0000 | **EXACT MATCH** |
| **5** | 3100 sqft, 5 beds, 4 baths, 2 yrs, 2.0 km | $711,693.19 | $711,693.19 | $0.0000 | **EXACT MATCH** |

---

## 11. Automated Test Suite Summary

Run the complete regression suite:
```bash
pytest tests/ -v
```
**Results: 9 passed in 3.35s (100% pass rate)**
* `test_data_loading_and_schema` (PASSED)
* `test_split_data_no_leakage` (PASSED)
* `test_pipeline_prediction_shape_and_type` (PASSED)
* `test_prediction_input_validation` (PASSED)
* `test_health_endpoint` (PASSED)
* `test_predict_valid_payload` (PASSED)
* `test_predict_missing_required_field` (PASSED)
* `test_predict_wrong_data_type` (PASSED)
* `test_predict_invalid_business_constraints` (PASSED)
