# Predicting Patient Admission

## Project Overview

This project builds a machine learning classifier to predict patient admission status ("IN" for inpatient, "OUT" otherwise) based on routine lab tests. The goal is to support hospital triage by flagging potential inpatients early, with a focus on maximizing recall to reduce false negatives that could delay care.

Docker makes sure the ML pipeline runs the same way every time by putting everything it needs like code, libraries, and settings into one container. This avoids issues from different environments. Airflow helps run each step of the pipeline in order. It can restart failed steps, keep track of progress, and handle more complex workflows, making the whole system easier to manage and scale.

MLflow is integrated for experiment tracking, model artifact storage, metric logging, and model registration. Evidently AI is used for data drift detection to ensure model reliability over time.



---

## Folder Structure

```
.
├─ deploy/
│  └─ docker/
│     ├─ Dockerfile.pipeline     # builds custom Airflow runtime (with your deps)
│     └─ Dockerfile.mlflow       # MLflow image (optional; used by compose)
├─ deploy/airflow/
│  ├─ dags/                      # your DAGs
│  ├─ logs/                      # runtime logs (mounted)
│  ├─ plugins/                   # custom operators/plugins
│  └─ config/                    # airflow.cfg (created by init)
├─ src/                          # pipeline code (imported by DAGs)
├─ data/                         # datasets (mounted)
├─ models/                       # serialized models/artifacts
├─ reports/                      # generated reports
├─ docker-compose.yaml
├─ pyproject.toml
├─ uv.lock                       # if you’re using uv lockfile
└─ .env                          # compose variables (you create this)

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
- **Docker.mlflow** extends the ML image to run in Apache Airflow for orchestration.

## 1 Create Docker Image

### 1.1 Build the ML Pipeline base image
```bash
docker compose -f docker-compose.yaml up -d --build
```
docker compose build --no-cache --progress=plain
docker compose run --rm airflow-init
docker compose up -d

### 1.2 Build the MLflow server image
```bash
docker compose run --rm airflow-init
```

### 1.3 Access the Airflow and MLflow:
- Access the Airflow UI at: http://localhost:8081
	- Login credentials (default):
	**Username**: airflow
	**Password**: airflow
- Access the MLflow UI at: http://localhost:5000

### 1.4 Confirm DAG is visible
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
## Decision Threshold for Inpatient Recall

By default, sklearn classifiers use a **0.50 cutoff** on the predicted probability of the positive class (`p(IN)`):
- `p(IN) ≥ 0.50` → predict **IN**
- `p(IN) < 0.50` → predict **OUT**

This maximizes accuracy but can miss too many inpatients (low recall).
Our business requirement is to **prioritize recall** for inpatients, so we adjust the threshold.

### How it works
We use `_select_threshold_for_recall(y_true, prob_pos, target_recall)`:
1. Sweep through possible cutoffs on the validation set.
2. Pick the **highest threshold** `t*` with recall ≥ target (e.g. 0.90).
3. Save `t*` to `mlflow/artifacts/models_export/threshold.txt` so it’s versioned and reused at inference.

### Example (illustrative)
| Policy                  | Threshold | Recall (IN) | Precision | Accuracy |
|--------------------------|-----------|-------------|-----------|----------|
| Default sklearn (0.50)   | 0.50      | 0.76        | 0.68      | 0.82     |
| **Recall-targeted**      | **0.31**  | **0.90**    | 0.52      | 0.78     |

Lowering the cutoff increases recall from **0.76 → 0.90**, meeting the requirement of catching ≥90% of inpatients.

### Rubric Alignment
- **Save model artifacts to `mlflow/artifacts/`** ✅
  `threshold.txt` is logged as an artifact to record the exact cutoff used.
- **Log model with custom PyFunc wrapper** ✅
  Wrapper returns labels; the saved threshold ensures predictions follow the chosen recall policy.
- Hyperparameters remain separate (we still log exactly 3 per model type).


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

## Reflection (August 20, 2025)

The biggest challenge that i faced what not being able to to launch airflow again. i thought i got this fixed in HW2. but seems need to start again for HW3 with the mlflow in the mix. when
---
