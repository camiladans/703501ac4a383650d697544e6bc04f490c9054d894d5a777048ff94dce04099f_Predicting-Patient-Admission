# Predicting Patient Admission

## Project Overview

This project builds a machine learning classifier to predict patient admission status ("IN" for inpatient, "OUT" otherwise) based on routine lab tests. The goal is to support hospital triage by flagging potential inpatients early, with a focus on maximizing recall to reduce false negatives that could delay care.

Docker ensures environment consistency with immutable infrastructure, while Airflow enables scalable orchestration through task retries and modular execution.

---

## Folder Structure

```
├── deploy/
│   ├── docker/               # Dockerfile and build logic
│   ├── airflow/
│   │   ├── config/           # Deployment configs
│   │   ├── dags/             # Airflow DAG definitions
│   │   ├── logs/             # Airflow task logs
│   │   ├── plugins/
│   └── config/
├── data/                     # Raw and processed input files
├── models/                   # Trained model outputs
├── reports/                  # Evaluation results
├── src/                      # Modular ML pipeline code
├── docker-compose.yml        # Airflow + Docker orchestration

```

---

## Setup Instructions

### Local Python + `uv` Setup (Optional for Development)

If you'd like to run the pipeline locally without Docker:

```bash
# 1. Install Python 3.12.8 via pyenv
curl https://pyenv.run | bash
pyenv install 3.12.8
pyenv local 3.12.8

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install uv via pipx
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install uv

# 4. Install dependencies from pyproject.toml + uv.lock
uv pip install --system

---

## Docker Setup

This project uses Docker to containerize the ML pipeline, ensuring repeatable and isolated execution.

### Build the Docker Image

```bash
docker build -t 703501ac4a383650d697544e6bc04f490c9054d894d5a777048ff94dce04099f-ml-pipeline -f deploy/docker/Dockerfile .
```

### Run the Pipeline Standalone (Directly via Docker)

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  703501ac4a383650d697544e6bc04f490c9054d894d5a777048ff94dce04099f-ml-pipeline
```

---

## Airflow Orchestration

Apache Airflow runs the ML pipeline through a defined DAG.

- **DAG file**: `deploy/airflow/dags/ml_pipeline_dag.py`
- **Trigger**: Manual (no schedule interval)

### DAG Tasks

1. `preprocess`: Clean and split data
2. `engineer`: Feature generation
3. `train`: Model training and save
4. `evaluate`: Metrics and visualizations

Run Airflow:

```bash
docker compose up --build
```

Then access the UI at:

```
http://localhost:8080
```

To test a task manually:

```bash
docker compose exec airflow-webserver airflow tasks test ml_pipeline_dag preprocess 2025-01-01
```

Logs will be available at `deploy/airflow/logs/`.

---

## Volume Mapping in Docker Compose

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

---

## Pre-commit Configuration

### Installed Hooks

- `ruff`: Python linter for style and unused code cleanup
- `end-of-file-fixer`: Ensures files end with one newline
- `trailing-whitespace`: Removes spaces/tabs at line ends
- `hadolint`: Lints `Dockerfile` for security/best practices
- `yamllint`: Validates `docker-compose.yml` formatting

### Usage

```bash
pre-commit install
pre-commit run --all-files
```

---

## .dockerignore Highlights

To keep images lean and secure:

```
__pycache__/
*.pyc
.venv/
data/
models/
reports/
.git/
```

---

## Reflection

Key challenges included adapting to modular pipeline code and managing `uv` dependency resolution inside Docker. SHAP failed to install via Airflow container due to lack of system-level support, requiring fallback to traditional model evaluation. Pre-commit hooks also occasionally caused formatting issues that needed manual resolution.

---
