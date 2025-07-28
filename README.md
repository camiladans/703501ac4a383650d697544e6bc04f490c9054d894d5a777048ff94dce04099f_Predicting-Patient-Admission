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

## Reflection

One major challenge I encountered was related to environment and code organization. Since the project used uv for dependency management, I initially tried uv add to include packages, but some installations failed. I had to fall back to using uv pip install and then manually update the pyproject.toml file to reflect the changes.

Another tricky issue was adapting to a modular ML project layout, where each pipeline stage is in a separate script. This made it harder to debug variable flow and track data transformations. Additionally, pre-commit formatting occasionally introduced inconsistent tab/space indentation, which led to errors like IndentationError: unindent does not match any outer indentation level. I had to manually fix these formatting issues and double-check indentation across all files.
