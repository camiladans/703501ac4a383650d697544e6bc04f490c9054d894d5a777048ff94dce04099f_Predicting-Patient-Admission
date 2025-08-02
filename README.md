# Predicting Patient Admission

## Project Overview

This project builds a machine learning classifier to predict patient admission status ("IN" for inpatient, "OUT" otherwise) based on routine lab tests. The goal is to support hospital triage by flagging potential inpatients early, with a focus on maximizing recall to reduce false negatives that could delay care.

Docker makes sure the ML pipeline runs the same way every time by putting everything it needs like code, libraries, and settings into one container. This avoids issues from different environments. Airflow helps run each step of the pipeline in order. It can restart failed steps, keep track of progress, and handle more complex workflows, making the whole system easier to manage and scale.

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

### Prerequisites
- Docker Desktop installed and running
- Python 3.12 (inside containers only)
- Airflow 3.0.3 (running in Docker)

This project uses a two-stage Docker setup:
- **Docker.pipeline** builds the ML pipeline image with all dependencies and source code.
- **Docker.airflow** extends the ML image to run in Apache Airflow for orchestration.

### 1. Build the ML Pipeline Image
This step builds the base image containing:
- All Python dependencies from pyproject.toml + uv.lock
- Source code from src/
```bash
docker build --no-cache -t 703501ac4a383650d697544e6bc04f490c9054d894d5a777048ff94dce04099f-ml-pipeline -f deploy/docker/Dockerfile.pipeline .
```

### 2. Build the Airflow Runtime Image
This adds Airflow bootstrapping layers on top of the image.
```bash
docker build -t custom-airflow:runtime -f deploy/docker/Docker.airflow .
```

### 3. Launch Airflow (Docker Compose)
- Initialize the Airflow metadata database.
```bash
docker compose -f deploy/docker-compose.yaml up airflow-init
```
- Then launch all Airflow services.
```bash
docker compose -f deploy/docker-compose.yaml up
```
- Access the Airflow UI at:
```bash
http://localhost:8080
```
- Login credentials (default):
**Username**: airflow
**Password**: airflow


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
