# Predicting Patient Admission (In vs Out)

## Project Overview

This project builds a machine learning classifier to predict patient admission status ("IN" for inpatient, "OUT" otherwise) based on routine lab tests. The goal is to support hospital triage by flagging potential inpatients early, with a focus on maximizing recall to reduce false negatives that could delay care.

Docker makes sure the ML pipeline runs the same way every time by putting everything it needs like code, libraries, and settings into one container. This avoids issues from different environments. Airflow helps run each step of the pipeline in order. It can restart failed steps, keep track of progress, and handle more complex workflows, making the whole system easier to manage and scale.

MLflow is integrated for experiment tracking, model artifact storage, metric logging, and model registration. Evidently AI is used for data drift detection to ensure model reliability over time.

---

## Setup Instructions

### 1. Build images

```bash
docker compose build --no-cache
```

### 2. Start services

```bash
docker compose up -d
```

### 3. Initialize Airflow

```bash
docker compose run airflow-init
```

### 4. Verify

* Airflow API: [http://localhost:8080](http://localhost:8080)
* MLflow UI: [http://localhost:5000](http://localhost:5000)

---

## MLflow Integration

* **Tracking URI**: `http://mlflow:5000`
* **Experiment**: `patient_experiments_v3`
* Logs model artifacts, metrics, and chosen thresholds.
* Models are registered in MLflow Model Registry.

---

## Model Drift Detection

The drift was simulated during preprocessing by modifying numeric features with Gaussian noise or scaling, and flipping 10–15% of categorical values to alternative classes. This produces `drifted_train.csv` and `drifted_test.csv`. Drift detection computes **PSI (Population Stability Index)** for up to three features, averages them into an `overall_drift_score`, and flags drift if any exceed 0.2. A slim JSON report (`reports/drift_report.json`) is saved and used by Airflow’s branching operator.

---

## Folder Structure

```text
├── deploy/
│   ├── docker/               # Dockerfiles and build logic for pipeline + MLflow
│   │   ├── Dockerfile.pipeline
│   │   ├── Dockerfile.mlflow
│   ├── airflow/              # Airflow runtime configuration
│   │   ├── config/           # Airflow configs (airflow.cfg, env vars)
│   │   ├── dags/             # DAG definitions (ml_pipeline_dag.py)
│   │   ├── logs/             # Task logs (mounted at runtime)
│   │   └── plugins/
├── src/                      # All project modules
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── drift_detection.py
│   ├── threshold_sweep.py
│   └── run_pipeline.py       # Local runner (outside Airflow)
├── data/
│   ├── raw/                  # Input raw data (data-ori.csv)
│   ├── train.csv, test.csv
│   ├── drifted_train.csv, drifted_test.csv
│   ├── train_fe.csv, test_fe.csv
├── reports/                  # Outputs and diagnostics
│   ├── drift_report.json
│   ├── evaluation_results.json
│   ├── threshold_sweep.csv
│   └── performance_check.json
├── models/                   # Exported models
├── mlflow/                   # MLflow artifacts + backend store
│   ├── artifacts/
│   └── runs/
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Locked dependency versions
├── docker-compose.yaml       # Orchestration for all services
└── README.md                 # Project documentation
```

This structure ensures clean separation: **src/** for code, **data/** for inputs, **reports/** for outputs, **mlflow/** for tracking, and **deploy/** for all container/orchestration logic.

---

## Testing Instructions

### Run locally

```bash
export MLFLOW_URI="file:./mlruns"
python -m src.run_pipeline
```

### Run inside Airflow

```bash
docker compose exec airflow-scheduler airflow dags trigger ml_pipeline_dag
```

### Inspect outputs

* `reports/evaluation_results.json` → accuracy + recall
* `reports/drift_report.json` → drift detection result
* MLflow UI for metrics and artifacts

---

## Reflection

Challenges included handling **drift detection without SciPy**, ensuring compatibility with **Airflow’s TaskFlow API**. Containerization required careful dependency pinning (`Werkzeug<3`, `Flask==2.2.5`) to align Airflow and Evidently. These adjustments made the pipeline reproducible, evaluable, and extensible for both local and orchestrated runs.

---
