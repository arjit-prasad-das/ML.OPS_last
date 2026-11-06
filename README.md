# MLOps Lab: Modular ML Pipelines, FastAPI & Docker (Sessions 7, 8, 9)

**Student Name:** Arjit Prasad Das  
**Roll Number:** 24AI013  
**Course:** B.Tech, Artificial Intelligence & Data Science  
**Subject:** MLOps: From Machine Learning to Production AI Systems  

---

## 1. Project Overview & Architecture

This repository transforms an existing Machine Learning model for **House Price Prediction** into a reproducible, production-ready, modular service exposed via **FastAPI** and packaged with **Docker** and **Docker Compose**.

### End-to-End Workflow Architecture

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

## 2. Directory Structure

```text
mlops-lab-7-9/
├── configs/
│   └── config.yaml              # Centralized YAML configuration for dataset, hyperparameters & paths
├── data/
│   ├── raw/
│   │   └── dataset.csv          # Versioned raw house price dataset (240 samples, 5 features + price)
│   └── processed/
│       ├── train.csv            # Cleaned training split (80%)
│       └── test.csv             # Cleaned held-out test split (20%)
├── models/
│   └── house_price_pipeline.joblib # Serialized end-to-end Scikit-learn inference pipeline
├── reports/
│   ├── metrics.json             # Held-out evaluation metrics (MAE, RMSE, R²)
│   ├── sample_predictions.json  # Predictions generated for 5 sample records
│   ├── prediction_comparison.json # Local vs Docker container prediction equivalence table
│   └── figures/
│       └── actual_vs_predicted.png # Regression scatter plot
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

## 3. Dataset & Problem Specification

* **Domain:** Real Estate Valuation / House Price Estimation (Regression).
* **Dataset:** Deterministic synthetic dataset consisting of 240 records with no personally identifiable information (PII).
* **Features:**
  1. `area_sqft` (float): Total floor space in square feet.
  2. `bedrooms` (integer): Number of bedrooms (range: 1–6).
  3. `bathrooms` (integer): Number of bathrooms (range: 1–4).
  4. `age_years` (integer): Age of property in years (range: 0–60).
  5. `distance_km` (float): Distance to metropolitan center in km (range: 0.5–25.0).
* **Target:** `price` (float): Property value in USD.
* **Pipeline Estimator:** `StandardScaler` followed by `RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)`.

---

## 4. Setup & Installation

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone <YOUR_REPO_URL>
cd mlops-lab-7-9

# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS:
source .venv/bin/activate

# Install locked dependencies:
pip install -r requirements.txt
```

---

## 5. Execution Guide: Part A — ML Pipelines

### 1. Run Pipeline Training from Terminal
```bash
python -m src.train
```
*Loads raw data, performs schema validation, splits train/test partitions without leakage, fits the unified pipeline, and saves `models/house_price_pipeline.joblib`.*

### 2. Run Pipeline Evaluation
```bash
python -m src.evaluate
```
*Scores held-out test set (`data/processed/test.csv`), generates `reports/metrics.json`, and renders `reports/figures/actual_vs_predicted.png`.*

### 3. Generate Predictions on Sample Records
```bash
python -m src.predict
```
*Scores 5 records and stores results in `reports/sample_predictions.json`.*

### 4. Execute Automated Pipeline Tests
```bash
pytest tests/test_pipeline.py -v
```

---

## 6. Execution Guide: Part B — FastAPI REST Service

### 1. Launch FastAPI Server Locally
```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```
Interactive Swagger UI Documentation is accessible at:  
👉 **http://127.0.0.1:8000/docs**

### 2. Test Endpoints

#### A. Health Check (`GET /health`)
```bash
curl -X GET http://127.0.0.1:8000/health
```
**Response JSON (200 OK):**
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
**Response JSON (200 OK):**
```json
{
  "predicted_price": 423884.13,
  "currency": "USD",
  "status": "success"
}
```

#### C. Validation Testing: Missing Field (Expected 422)
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area_sqft": 1800.0,
    "bedrooms": 3,
    "bathrooms": 2,
    "age_years": 5
  }'
```
**Response JSON (422 Unprocessable Entity):**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "distance_km"],
      "msg": "Field required"
    }
  ]
}
```

#### D. Validation Testing: Invalid Data Type (Expected 422)
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area_sqft": "one thousand",
    "bedrooms": 3,
    "bathrooms": 2,
    "age_years": 5,
    "distance_km": 12.0
  }'
```
**Response JSON (422 Unprocessable Entity):**
```json
{
  "detail": [
    {
      "type": "float_parsing",
      "loc": ["body", "area_sqft"],
      "msg": "Input should be a valid number, unable to parse string as a number"
    }
  ]
}
```

### 3. Run Automated API Tests
```bash
pytest tests/test_api.py -v
```

---

## 7. Execution Guide: Part C — Dockerization

### 1. Build the Docker Image
```bash
docker build -t ml-inference-api:1.0 .
```

### 2. Verify Docker Image
```bash
docker image ls | grep ml-inference-api
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

### 5. Test Inference Against Docker Container
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area_sqft": 1800.0,
    "bedrooms": 3,
    "bathrooms": 2,
    "age_years": 5,
    "distance_km": 12.0
  }'
```

### 6. Stop and Clean Up Container
```bash
docker stop ml-api
docker rm ml-api
```

### 7. Multi-Service Architecture with Docker Compose
To run both the **FastAPI prediction service** and an **MLflow Tracking Server** simultaneously:
```bash
# Validate compose syntax
docker compose config

# Build and start services in background
docker compose up --build -d

# Check running services
docker compose ps

# View aggregate logs
docker compose logs -f

# Teardown services and networks
docker compose down
```

---

## 8. Part D — Final Integration & Verification

To verify numerical parity between the local Python environment and the Docker containerized API:
```bash
python src/verify_predictions.py
```
This tests 5 sample records simultaneously against the local `Pipeline.predict()` and the container's HTTP endpoint `http://localhost:8000/predict`, validating that:
$$\Delta = |y_{\text{local}} - y_{\text{docker}}| \le 0.01$$

---

## 9. Architecture Analysis: Online vs. Batch Inference Design (B6)

When scaling from individual request scoring to scoring large volumes (e.g., millions of records via CSV uploads):

1. **Online Synchronous Endpoints:**
   * Best for real-time web/mobile applications requiring sub-100ms latency.
   * Drawback: Long-running HTTP requests timing out (504 Gateway Timeout), memory bottlenecks when parsing large files in RAM.
2. **Asynchronous Background Processing (Recommended Batch Strategy):**
   * Endpoint `POST /predict/batch` receives CSV file, validates schema, immediately returns a `job_id` with `202 Accepted`.
   * Work is dispatched to Celery/Redis or an asynchronous worker queue.
   * Worker processes records in chunks (e.g., using `pandas.read_csv(..., chunksize=10000)`), streams predictions to cloud storage (S3/GCS) or a database, and updates status.
   * Client polls `GET /jobs/{job_id}` or receives a Webhook callback upon completion.

---

## 10. Summary of Automated Tests

Execute the complete regression test suite:
```bash
pytest tests/ -v
```
* **Test 1:** `test_data_loading_and_schema` — Ensures dataset integrity and schema compliance.
* **Test 2:** `test_split_data_no_leakage` — Asserts train and test splits have 0 index overlap.
* **Test 3:** `test_pipeline_prediction_shape_and_type` — Asserts model output shape and non-negative values.
* **Test 4:** `test_prediction_input_validation` — Asserts missing columns raise `ValueError`.
* **Test 5:** `test_health_endpoint` — Asserts `/health` reports 200 OK and model ready.
* **Test 6:** `test_predict_valid_payload` — Asserts `/predict` yields valid price estimation.
* **Test 7:** `test_predict_missing_required_field` — Asserts missing payload fields return 422.
* **Test 8:** `test_predict_wrong_data_type` — Asserts type mismatch returns 422.
* **Test 9:** `test_predict_invalid_business_constraints` — Asserts negative dimensions return 422.
