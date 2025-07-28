# 703501ac4a383650d697544e6bc04f490c9054d894d5a777048ff94dce04099f_Predicting Patient Admission
Predicting Patient Admission

# Predicting Patient Admission

## Project Overview
This project aims to build a machine learning classification model to predict whether a patient will be admitted as an inpatient ("IN") or not ("OUT") based on routine laboratory test results. The goal is to assist hospitals in early triage by identifying potential inpatients from lab results alone. We prioritized recall of inpatients to minimize the risk of false negatives, which in a clinical setting could lead to critical delays in care.

## Folder Structure
	├── deploy/					 # Scripts for orchestrating deployment
	│   ├── docker/              # Dockerfiles and build artifacts
	│   ├── airflow/			 # Specific Airflow container setup
	│   │   ├── dags/            # Contains Airflow DAG scripts that define the ML workflow
	│   │   └── logs/            # Holds log outputs from Airflow tasks
	│   └── config/              # Airflow scheduler and deployment configs
	|── data/
	│ ├── raw/ # Original .xlsx input files
	├── models/ # Trained model outputs
	├── reports/ # Evaluation metrics (e.g., accuracy)
	├── src/ # Modular code by ML lifecycle
	├── main.py # Entrypoint script
	├── docker-compose.yml		 # Defines services like Airflow scheduler, webserver, and Postgres DB


This folder structure keeps data in data/raw folder. All codes are in the src folder with separate folders for the trained model output and reports to help organize and make results reproducible and easy to manage.

Isolating DAGs in `deploy/airflow/dags/` ensures modularity, allowing independent testing of workflow tasks without affecting the core ML code in `src/`. The separation of `docker/` and `config/` improves maintainability and scalability of deployment pipelines.




This modular structure supports containerized and orchestrated workflows by isolating DAGs and build logic from core ML code, ensuring maintainability and scalability.


## Setup Instructions
1- Install pyenv

	If pyenv is not installed, run:
	curl https://pyenv.run | bash

	Then add the following to your ~/.zshrc (or ~/.bashrc):
	export PYENV_ROOT="$HOME/.pyenv"
	export PATH="$PYENV_ROOT/bin:$PATH"
	eval "$(pyenv init --path)"
	eval "$(pyenv init -)"

	Reload your terminal:
	source ~/.zshrc

2- Install Python 3.12.8

	bash:
	pyenv install 3.12.8
	pyenv local 3.12.8

	Check that it worked:
	python --version  # Should show Python 3.12.8

3- Create and activate a virtual environment

	bash:
	python -m venv .venv
	source .venv/bin/activate

4- Install uv (via pipx)

	bash:
	python3 -m pip install --user pipx
	python3 -m pipx ensurepath
	pipx install uv

5- Initialize the project with uv

	bash: uv init
	Update in the prompts (project name, version, etc.)

6- Install required dependencies

	bash:
	uv pip install \
	numpy pandas matplotlib seaborn tqdm \
	scikit-learn xgboost imbalanced-learn \
	pre-commit

## How to setup Docker and Airflow

1- Download and install Docker desktop
Reference: https://www.docker.com/

2- Download airflow docker compose file and save to the project

## Pre-Commit Configuration
The pre-commit hooks help enforce PEP 8 by using ruff, which checks code against PEP 8 standards and common linting issues. Additionally, end-of-file-fixer and trailing-whitespace ensure clean formatting by enforcing newline rules and removing unnecessary whitespace, both of which align with PEP 8 guidelines.

Pre-commit hooks used include:
1- ruff:

	This is a Python linter and formatter. It ensures consistent code style, removes unused imports, sorts imports properly, and catches common coding errors before they cause bugs.

2- end-of-file-fixer:

	This ensures every file ends with exactly one newline character. This is a POSIX standard and prevents unnecessary changes in version control diffs.

3_trailing-whitespace:

	Detects and removes spaces or tabs at the end of lines.

How to use the Pre-commit?

	bash:
	pre-commit install
	pre-commit run --all-files

## .dockerignore Configuration

To improve Docker build performance and prevent unnecessary files from being copied into the container, we’ve added a `.dockerignore` file to the project root.

### Purpose
This file ensures that large or sensitive files—such as local data, virtual environments, system artifacts, and Git history—are excluded from the Docker build context.

### Excluded Files and Folders

- Python cache files: `__pycache__/`, `*.pyc`, `*.pyo`, `*.egg-info/`
- Jupyter artifacts: `.ipynb_checkpoints/`
- Data directories (mounted at runtime): `data/`, `models/`, `reports/`
- Virtual environments: `.venv/`
- System/editor files: `.DS_Store`, `.env`, `*.log`
- Git metadata: `.git/`, `.gitignore`

Keeping the Docker image clean helps reduce build time, improves security, and avoids bloated containers.

## 🐳 Docker Setup

This project includes a Docker setup to ensure reproducible, environment-agnostic execution of the ML pipeline.

### Dockerfile Overview

- **Base Image**: Uses `python:3.12-slim` for a minimal, fast Python runtime.
- **Dependency Management**: Installs Python packages using `uv` for fast, deterministic builds via `pyproject.toml`.
- **Working Directory**: Set to `/app` to isolate pipeline code.
- **Source Code**: Copies all pipeline scripts from the `src/` directory.
- **Entrypoint**: Runs `src/run_pipeline.py` automatically when the container starts.

### Building the Image

From the project root, run:

```bash
docker build -t 703501ac4a383650d697544e6bc04f490c9054d894d5a777048ff94dce04099f-ml-pipeline -f deploy/docker/Dockerfile .

## Airflow Orchestration

This project uses Apache Airflow to orchestrate the ML pipeline. Tasks like preprocessing, feature engineering, model training, and evaluation are defined as `PythonOperator` steps in a DAG.

**DAG File**: `deploy/airflow/dags/ml_pipeline_dag.py`
**Schedule**: Manually triggered (`schedule_interval=None`)

### DAG Structure

1. `preprocess`: Reads and splits the data
2. `engineer`: Performs feature engineering
3. `train`: Trains and saves the model
4. `evaluate`: Generates evaluation metrics and saves outputs

Task dependencies are defined sequentially. The DAG can be triggered in the Airflow UI at [http://localhost:8080](http://localhost:8080).

---

## Docker Integration

### Dockerfile Strategy

- Uses `python:3.12-slim` as a minimal base image.
- Installs dependencies using `uv` (via `pipx`) for deterministic builds.
- Reads from `pyproject.toml` and `uv.lock`.
- Mounts `src/`, `data/`, and `models/` via volumes to persist across containers.

### Build the Docker Image

```bash
docker build -t 703501ac4a383650d697544e6bc04f490c9054d894d5a777048ff94dce04099f-ml-pipeline -f deploy/docker/Dockerfile .
```

---

## Running Airflow with Docker

1. Start the environment:

```bash
docker compose up --build
```

2. Access Airflow UI:

```
http://localhost:8080
```

3. Trigger the DAG manually in the UI, or test individual tasks:

```bash
docker compose exec airflow-webserver airflow tasks test ml_pipeline_dag preprocess 2025-01-01
```

4. Logs will appear in: `deploy/airflow/logs/`

---

## Docker Compose Volumes

Example volume configuration in `docker-compose.yml`:

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

## Pre-Commit Hooks for Docker and Airflow

To improve code quality and container security, additional pre-commit hooks have been added.

### New Hooks

- `hadolint`: Lints `Dockerfile` for best practices and common issues like using `root`, missing `CMD`, or insecure `apt` usage.
- `yamllint`: Checks formatting and syntax for YAML files like `docker-compose.yml`.

### Usage

```bash
pre-commit install
pre-commit run --all-files
```

## Reflection: HW1

One major challenge I encountered was related to environment and code organization. Since the project used uv for dependency management, I initially tried uv add to include packages, but some installations failed. I had to fall back to using uv pip install and then manually update the pyproject.toml file to reflect the changes.

Another tricky issue was adapting to a modular ML project layout, where each pipeline stage is in a separate script. This made it harder to debug variable flow and track data transformations. Additionally, pre-commit formatting occasionally introduced inconsistent tab/space indentation, which led to errors like IndentationError: unindent does not match any outer indentation level. I had to manually fix these formatting issues and double-check indentation across all files.
