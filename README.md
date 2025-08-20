# Predicting Patient Admission

## Project Overview

This project builds a machine learning classifier to predict patient admission status ("IN" for inpatient, "OUT" otherwise) based on routine lab tests. The goal is to support hospital triage by flagging potential inpatients early, with a focus on maximizing recall to reduce false negatives that could delay care.

Docker makes sure the ML pipeline runs the same way every time by putting everything it needs like code, libraries, and settings into one container. This avoids issues from different environments. Airflow helps run each step of the pipeline in order. It can restart failed steps, keep track of progress, and handle more complex workflows, making the whole system easier to manage and scale.

MLflow is integrated for experiment tracking, model artifact storage, metric logging, and model registration. Evidently AI is used for data drift detection to ensure model reliability over time.

---

## Folder Structure

```
├── deploy/
│   ├── docker/                      # Dockerfiles (pipeline/airflow)
│   └── docker-compose.yaml          # Orchestration for Airflow + MLflow
├── data/
│   ├── raw/                         # Canonical raw data (ingestion writes here)
│   ├── train.csv
│   ├── test.csv
│   ├── drifted_train.csv
│   ├── drifted_test.csv
│   ├── drifted_train_cycle{2..N}.csv
│   └── drifted_test_cycle{2..N}.csv
├── mlflow/                          # Local tracking storage (volume/bind)
│   ├── runs/                        # Populated by MLflow tracking server
│   └── artifacts/                   # Model and other artifacts
├── models/                          # Extra saved models (optional)
├── reports/
│   ├── evaluation_results.json
│   ├── performance_check.json
│   ├── drift_report.json            # (if your drift module writes one)
│   └── test_feature_drifts.json     # per-cycle feature drifts (logged)
├── src/
│   ├── data_ingestion.py            # ingest_data(source_path) -> data/raw/data-ori.csv
│   ├── data_preprocessing.py        # preprocess_data(...) -> returns X/y + drifted X/y
│   ├── feature_engineering.py
│   ├── model_training.py            # train_model(...), MLflow logging in training
│   ├── evaluation.py                # evaluate_model(...): logs accuracy & f1_score
│   ├── drift_detection.py           # detect_drift(reference_csv, current_csv)
│   └── run_pipeline.py              # end-to-end pipeline with MLflow + drift cycles
├── .pre-commit-config.yaml
├── .dockerignore
└── README.md

```

---

## Setup Instructions

### Prerequisites
- Docker Desktop installed and running
- Python 3.12 (inside containers only)
- Airflow 3.0.3 (running in Docker)
- MLflow server available at http://localhost:5000 (via Docker Compose)

This project uses a two-stage Docker setup:
- **Docker.pipeline** builds the ML pipeline image with all dependencies and source code.
- **Docker.airflow** extends the ML image to run in Apache Airflow for orchestration.

## 1 Create Docker Image and Launch Airflow using docker compose

### 1.1 Build the ML Pipeline base image and Airflow Runtime on top of it
```bash
docker build --no-cache -t 703501ac4a383650d697544e6bc04f490c9054d894d5a777048ff94dce04099f-ml-pipeline -f deploy/docker/Dockerfile.pipeline .
```
- Based on apache/airflow:3.0.3
- Installs all deps from pyproject.toml + uv.lock
- Copies source code from src/

<!-- ### 1.2 Build the Airflow Runtime image (scheduler/webserver/etc.)
```bash
docker build --no-cache -t custom-airflow:runtime \
  -f deploy/docker/Dockerfile.airflow .
```
- Extends the ML pipeline base image
- Adds Airflow entrypoint and runtime configuration
- Used by docker-compose.yaml for all Airflow services -->

### 1.3 Build the MLflow server image
```bash
docker build --no-cache -t 703501ac4a383650d697544e6bc04f490c9054d894d5a777048ff94dce04099f_predicting-patient-admission-mlflow \
  -f deploy/docker/Dockerfile.mlflow .
```

### 1.4 Bring up Infra services only
```bash
docker compose -f docker-compose.yaml up -d \
  postgres mlflow
```
- Check if they are healthy:
```bash
docker compose -f docker-compose.yaml ps
```

### 1.5 Launch Airflow DB (Initialize)
```bash
export AIRFLOW_UID=$(id -u)
docker compose -f docker-compose.yaml up airflow-init
```
Run this:
- If running for the first time, initialize the Airflow metadata database.
- If you wiped volumes with docker compose down -v.
- If you deliberately want to reinitialize the metadata DB.

For later runs, use this:
```bash
docker compose -f docker-compose.yaml up -d
```

### 1.6 Launch Airflow Services
```bash
docker compose -f docker-compose.yaml up -d \
  airflow-scheduler airflow-webserver airflow-dag-processor airflow-triggerer
```

### 1.7 Health Checks
- Scheduler logs (tail a bit, then Ctrl+C)
```bash
docker compose -f docker-compose.yaml logs -f airflow-scheduler | head -n 80
```

- DAG present?
```bash
docker compose -f docker-compose.yaml exec airflow-scheduler ls -l /opt/airflow/dags
```

- Source code (src) + raw data present?
```bash
docker compose -f docker-compose.yaml exec airflow-scheduler ls -l /app/src
docker compose -f docker-compose.yaml exec airflow-scheduler ls -l /app/data /app/data/raw
```

- Check mlflow
```bash
docker-compose up -d
curl http://localhost:5000 # Should return HTML
```

### 1.8 Access the Airflow and MLflow:
- Access the Airflow UI at: http://localhost:8080
	- Login credentials (default):
	**Username**: airflow
	**Password**: airflow
- Access the MLflow UI at: http://localhost:5000

### 1.9 Confirm DAG is visible
```bash
docker compose -f docker-compose.yaml exec airflow-scheduler airflow dags list | grep ml_pipeline_dag
```

## MLflow Integration
This project integrates **MLflow** for experiment tracking, model logging, and drift monitoring.

### Tracking Setup
- MLflow tracking server runs inside Docker on `http://localhost:5000`.
- The project sets the tracking URI in both `src/model_training.py` and `src/run_pipeline.py`:
  ```python
  	mlflow.set_tracking_uri("http://localhost:5000")
  ```
- Models, metrics, and artifacts are logged under:
  ```bash
  	mlflow/runs/
	mlflow/artifacts/
  ```

### Logged Parameters & Metrics
For classification models (our case: inpatient vs outpatient), the following are logged:
- Hyperparameters: n_estimators, max_depth, random_state
- Metrics: accuracy, f1_score
Artifacts (trained models) are saved to mlflow/artifacts/.

### Model Registration
After evaluation, the pipeline checks if performance meets thresholds:
- Classification: accuracy > 0.8 (configurable in src/drift_detection.py and src/run_pipeline.py)
  ```python
	mlflow.register_model(
		f"runs:/{mlflow.active_run().info.run_id}/model",
		"patient_admission_classifier"
	)
  ```

---

## Model Drift Detection

This project includes both **drift simulation** and **drift detection**.

### Drift Simulation
Robustness under distributional shift is evaluated by synthesizing drift across both numeric and categorical inputs. For numeric features, a 1.2× multiplicative shift or Gaussian noise with standard deviation fixed to 10% of the training standard deviation (𝜎_train) is applied. This combination represents both covariate scale changes (e.g., assay calibration drift) and random variability (e.g., acquisition noise). For categorical features, 10–15% of non-null values are flipped uniformly at random to a different class, simulating coding inconsistencies or evolving clinical processes. Drift is then detected using Evidently’s DataDriftPreset, producing both a dataset-level flag and feature-level contributions. Monitoring is performed twice per run on distinct drifted test files to emulate repeated checks over time.

The drifted versions are saved as:
data/drifted_train.csv
data/drifted_test.csv

**Justification**:
Simulating drift allows us to validate whether the pipeline can recognize shifts in feature distributions that mimic real-world changes (e.g., new patient demographics, different lab equipment calibrations, or evolving coding practices). Without such testing, a model might silently degrade in production. By systematically injecting noise and category flips, we approximate realistic feature drift scenarios and create a benchmark for evaluating Evidently’s detection accuracy.

### Model Drift Detection
src/drift_detection.py was added to monitor changes in input data over time.
- Tool: evidently
- Method: DataDriftPreset to detect dataset and feature-level drift
- Output: JSON report written to reports/drift_report.json

Example format:
	```json
		{
		"drift_detected": true,
		"feature_drifts": {
			"feature1": 0.12,
			"feature2": 0.34,
			"feature3": 0.07
		},
		"overall_drift_score": 0.18
		}
	```
The drift check runs after evaluation in run_pipeline.py.

## Airflow DAG Overview
The DAG is defined using the @dag decorator in ml_pipeline_dag.py. Each step in the ML pipeline (preprocessing, feature engineering, training, evaluation) is wrapped in an @task to enable modular, trackable execution.

The pipeline consists of four tasks:
1. `preprocess` - clean and split input data
2. `engineer` - Generate features for training
3. `train` - Train and save the classification model
4. `evaluate` - Generate metrics and visualizations

Setting **schedule=None means the DAG must be triggered manually** via the Airflow UI.


## Running the ML Pipeline in Airflow
The main DAG is defined in: `deploy/airflow/dags/ml_pipeline_dag.py`

In the Airflow UI:
- Enable the DAG named 'ml_pipeline_dag'
- Click the "Play Button" to trigger the DAG

Logs will be available at `deploy/airflow/logs/`.

To run a task manually (e.g., preprocess):
```bash
docker compose exec airflow-webserver airflow tasks test ml_pipeline_dag preprocess 2025-01-01
```

## Monitoring DAGs in Airflow UI
The Airflow web interface helps verify that the ML pipeline is running correctly from start to finish. It allows you to:
- View all available DAGs
- Manually trigger DAG runs
- Monitor task statuses (e.g., success, failed, retrying)
- Access detailed logs for debugging
- Visualize task dependencies and schedules


## Optional: Local Development without Docker
You can run the ML pipeline outside of Docker for faster debugging and testing.

Note: This setup is optional. Use only if you want to develop or test the pipeline locally without launching Docker or Airflow.

1. Install Python 3.12.8 via pyenv
```bash
curl https://pyenv.run | bash
pyenv install 3.12.8
pyenv local 3.12.8
```

2. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install uv via pipx
```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
exec $SHELL  # reload shell so 'uv' is on PATH
pipx install uv
```

4. Install project dependencies
```bash
uv pip install --system
```

Then run pipeline manually with:
```bash
python src/run_pipeline.py
```

---

## Development Notes

### Volume Mapping in Docker Compose

```yaml
volumes:
  - ./deploy/airflow/dags:/opt/airflow/dags
  - ./deploy/airflow/logs:/opt/airflow/logs
  - ./deploy/airflow/plugins:/opt/airflow/plugins
  - ./src:/app/src
  - ./data:/app/data
  - ./models:/app/models
  - ./reports:/app/reports
```
These mappings ensure that your local code and data are available inside the containers.

### Pre-commit Configuration

Installed Hooks:
- `ruff`: Python linter for style and unused code cleanup
- `end-of-file-fixer`: Ensures files end with one newline
- `trailing-whitespace`: Removes spaces/tabs at line ends
- `hadolint`: Lints `Dockerfile` for security/best practices
- `yamllint`: Validates `docker-compose.yml` formatting

Usage:
```bash
pre-commit install
pre-commit run --all-files
```

### .dockerignore Highlights

To keep images lean and secure:
```dockerignore
__pycache__/
*.pyc
.venv/
data/
models/
reports/
.git/
```

---

## Reflection (August 3, 2025)

One challenges I faced was getting SHAP and matplotlib to work inside the Docker image. Even though they were listed in my dependency files, the tools were not found after building the image. I tried troubleshooting the issue, but it took too much time and delayed progress. In the end, I decided to remove SHAP and matplotlib from the pipeline to focus on getting the core ML workflow running. Aside from that, setting up pre-commit hooks also caused a few formatting problems that I had to fix manually. Overall, adapting the modular pipeline code to work with Airflow and Docker took some trial and error, especially with managing the uv dependency setup.

---
