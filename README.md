# Predicting Patient Admission (Inpatient vs Outpatient)

## Overview

This project implements an **end-to-end ML pipeline** for predicting whether a patient admission is **Inpatient (IN)** or **Outpatient (OUT)**. The pipeline is containerized, orchestrated with Airflow, and integrates **MLflow** for experiment tracking, model registration, and artifact management. It includes **drift detection**, **branching retraining**, and a **threshold selection utility** to balance recall and precision.

---

## Setup Instructions

### 1. Build the Docker images

```bash
docker compose build
```

### 2. Start all services

```bash
docker compose up -d
```

### 3. Verify Airflow

Airflow API server should be running at [http://localhost:8080](http://localhost:8080).

Ensure `airflow-apiserver`, `airflow-scheduler`, `mlflow`, `postgres`, `postgres-mlflow`, and `redis` are healthy:
```bash
docker compose ps
```

Verify Airflow DAGs:
```bash
# List DAGs
docker compose exec -T airflow-scheduler airflow dags list

# Trigger DAG manually
docker compose exec -T airflow-scheduler airflow dags trigger ml_pipeline_dag

# Check logs
docker compose logs -f airflow-scheduler
```

### 4. Verify MLflow

MLflow UI should be available at [http://localhost:5000](http://localhost:5000).

```bash
curl http://localhost:5000/api/2.0/mlflow/experiments/list
```

---

## MLflow Integration

* **Tracking URI**: `http://mlflow:5000`
* **Experiment**: `patient_experiments_v3`
* **Artifacts**: stored under `mlflow/artifacts` (mounted into container)
* **Runs**: stored under `mlflow/runs`
* **PyFunc Models**: Logged with a custom wrapper ensuring spec compliance
* **Model Registry**: Automatically registers models if performance thresholds are met

Access via browser: [http://localhost:5000](http://localhost:5000)

---

## Core DAG Design

### Core Tasks

- preprocess_data — Ingests raw data, adds abnormal lab flags, splits train/test, and synthesizes drifted copies.
- feature_engineering — Builds numeric/derived features while keeping original columns.
- train_model — Trains recall-oriented RF/LogReg, logs exactly 3 hyperparameters and model artifacts to MLflow.
- evaluate_model — Loads the MLflow PyFunc model, evaluates accuracy + recall, and (if probs available) runs a threshold sweep.
- drift_detection — Uses PythonOperator to run drift check on data/test.csv vs data/drifted_test.csv, writing to reports/drift_report.json.

### Drift Detection Task
- Implemented with PythonOperator (task_id="drift_detection")
- Calls detect_drift("data/test.csv", "data/drifted_test.csv")
- Reads reports/drift_report.json (overwritten each run)

### Branching Logic
- BranchPythonOperator (task_id="branch_on_drift")
- Reads drift result from JSON file (not XCom)
- If drift_detected=True ➜ branch to retrain_model
- Else ➜ pipeline_complete

### End Tasks
- retrain_model — Re-runs the same training logic on original non-drifted data/train_fe.csv.
- pipeline_complete — Simple terminal marker for the success path.

### Dependencies
```text
preprocess_data >> feature_engineering >> train_model >> evaluate_model \
>> drift_detection >> branch_on_drift >> [retrain_model, pipeline_complete]
```

---

## Model Drift Detection

The **data drift** was simulated during preprocessing by deliberately updating both numeric and categorical features. Numeric features are either multiplied by a factor (×1.2) or injected with Gaussian noise proportional to training variance. Categorical features undergo random flipping (10–15% of values changed to another valid category). These drifted datasets (`drifted_train.csv`, `drifted_test.csv`) are compared against clean splits using **Population Stability Index (PSI)** and **Evidently’s DataDriftPreset**. Drift is flagged if any PSI > 0.2. This controlled simulation allows us to test retraining logic in a reproducible way, ensuring the pipeline can adapt to real-world changes in patient demographics or lab patterns.

---

## Evaluation Metrics

The metrics used were **accuracy** and **recall**. Recall is prioritized because missing an inpatient (false negative) is riskier than incorrectly predicting inpatient (false positive). Accuracy is included for overall balance. F1 score is not logged directly (per spec), but it is evaluated in threshold sweeps to balance trade-offs.

---

## Threshold Selection Justification

A custom `threshold_sweep.py` module runs multiple thresholds across probabilities, computing metrics for each. The system selects the **highest threshold that still achieves ≥0.90 recall**, ensuring sensitivity to inpatients. If no threshold reaches this floor, we fall back to the best threshold by F1 score, since it balances precision and recall.

---

## Folder Structure

```text
├── deploy/
│   ├── docker/                  # Dockerfiles for pipeline + MLflow
│   │   ├── Dockerfile.pipeline
│   │   ├── Dockerfile.mlflow
│   ├── airflow/                 # Airflow runtime configuration
│   │   ├── config/              # airflow.cfg and overrides
│   │   ├── dags/                # DAGs (ml_pipeline_dag.py)
│   │   ├── logs/                # Task logs (mounted)
│   │   ├── plugins/
│   └── config/                  # Project-level configs
├── src/                         # Pipeline source code
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── drift_detection.py
│   ├── threshold_sweep.py
│   └── run_pipeline.py
├── data/
│   ├── raw/                     # Raw source CSVs
│   ├── train.csv, test.csv
│   ├── drifted_train.csv, drifted_test.csv
│   ├── train_fe.csv, test_fe.csv
├── reports/                     # Generated reports
│   ├── drift_report.json
│   ├── evaluation_results.json
│   ├── threshold_sweep.csv
│   ├── performance_check.json
├── models/                      # Serialized model artifacts
├── mlflow/
│   ├── artifacts/               # Artifact store (logged by MLflow)
│   ├── runs/                    # Run metadata + checkpoints
├── uv.lock                      # Dependency lockfile
├── pyproject.toml               # Dependencies + metadata
├── docker-compose.yaml          # Orchestration of all services
```

### Mounting Behavior

* `/app/src` → Source code (importable by Airflow and run\_pipeline)
* `/opt/airflow/data` → Mounted local `./data/` for train/test splits
* `/opt/airflow/reports` → Mounted local `./reports/`
* `/mlflow/artifacts` → Persistent artifact store for MLflow server

This ensures consistent paths across Airflow, MLflow, and local development.

---

## Dependencies

Key pinned dependencies (as tested):

```toml
evidently == 0.7.11
mlflow == 3.1.4
psycopg2-binary == 2.9.10
```

Additional dependencies included to resolve compatibility issues during troubleshooting:

```toml
flask == 2.2.5
connexion == 2.14.2
werkzeug < 3
```
---

## Testing Instructions

### Smoke Test for Docker Compose

Bring up the stack
```bash
docker compose up --build
```

Verify the DAG is present
```bash
docker compose exec -T airflow-scheduler airflow dags list
```

Trigger the DAG
```bash
docker compose exec -T airflow-scheduler airflow dags trigger ml_pipeline_dag
```

Test a single task
```bash
docker compose exec -T airflow-scheduler \
  airflow tasks test ml_pipeline_dag preprocess_data 2025-08-25
```

Watch task logs
```bash
docker compose logs -f airflow-scheduler
```

Check artifacts & reports
```bash
# Evaluation metrics (accuracy, recall)
cat reports/evaluation_results.json

# Drift output (PSI & overall_drift_score)
cat reports/drift_report.json

# Threshold sweep (if evaluation had probs)
cat reports/threshold_sweep.csv
```

Confirm MLflow runs & artifacts byt opening UI: http://localhost:5000
You should see: runs under patient_experiments_v3, logged metrics, and the model artifact.

### Run end-to-end pipeline locally (no Airflow)

```bash
export MLFLOW_URI="file:./mlruns"
python -m src.run_pipeline
```

### Trigger Airflow DAG manually

```bash
docker compose exec -T airflow-scheduler \
  airflow dags trigger ml_pipeline_dag
```

### Check DAG task logs

```bash
docker compose logs -f airflow-scheduler
```

### Validate drift detection

```bash
cat reports/drift_report.json
```

---

## Reflection 25 Aug 2025

Working on this project surfaced two main challenges:

1. Working with the exact version of Evidently, I finally found the right list of imports that actually worked with version 0.7.11. This took a long time and required a lot of trial and error to stabilize the environment.
2. Understanding the mounting process of Airflow was confusing at first, how /app/src vs /opt/airflow vs /mlflow were mounted. I learned to refresh containers and remount volumes instead of rebuilding images every time. This sped up iteration a lot.
