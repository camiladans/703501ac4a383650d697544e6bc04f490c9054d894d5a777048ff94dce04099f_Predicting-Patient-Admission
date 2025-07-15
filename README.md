# 703501ac4a383650d697544e6bc04f490c9054d894d5a777048ff94dce04099f_Predicting Patient Admission
Predicting Patient Admission

# Predicting Patient Admission

## Folder Structure
├── data/
│ ├── raw/ # Original .xlsx input files
│ └── processed/ # Cleaned data
├── models/ # Trained model outputs
├── reports/ # Evaluation metrics (e.g., accuracy)
├── src/ # Modular code by ML lifecycle
├── main.py # Entrypoint script

This folder structure keeps raw data separate from processed outputs. All codes are in the src folder with separate folders for the trained model output and reports to help organize and make results reproducible and easy to manage.

## Setup Instructions
1_Install pyenv
	If pyenv is not installed, run:
	curl https://pyenv.run | bash

	Then add the following to your ~/.zshrc (or ~/.bashrc):
	export PYENV_ROOT="$HOME/.pyenv"
	export PATH="$PYENV_ROOT/bin:$PATH"
	eval "$(pyenv init --path)"
	eval "$(pyenv init -)"

	Reload your terminal:
	source ~/.zshrc

2_Install Python 3.12.8
	bash:
	pyenv install 3.12.8
	pyenv local 3.12.8

	Check that it worked:
	python --version  # Should show Python 3.12.8

3_Create and activate a virtual environment
	bash:
	python -m venv .venv
	source .venv/bin/activate

4_Install uv (via pipx)
	bash:
	python3 -m pip install --user pipx
	python3 -m pipx ensurepath
	pipx install uv

5_Initialize the project with uv
	bash: uv init
	Update in the prompts (project name, version, etc.)

6_Install required dependencies
	bash:
	uv pip install \
	numpy pandas matplotlib seaborn tqdm \
	scikit-learn xgboost imbalanced-learn \
	pre-commit

## Pre-Commit Configuration
The pre-commit hooks help enforce PEP 8 by using ruff, which checks code against PEP 8 standards and common linting issues. Additionally, end-of-file-fixer and trailing-whitespace ensure clean formatting by enforcing newline rules and removing unnecessary whitespace, both of which align with PEP 8 guidelines.

Pre-commit hooks used include:
1_ruff:
A fast Python linter and formatter. It ensures consistent code style, removes unused imports, sorts imports properly, and catches common coding errors before they cause bugs.

2_end-of-file-fixer:
Ensures every file ends with exactly one newline character. This is a POSIX standard and prevents unnecessary changes in version control diffs.

3_trailing-whitespace:
Detects and removes spaces or tabs at the end of lines.

How to use the Pre-commit?
	bash:
	pre-commit install
	pre-commit run --all-files
