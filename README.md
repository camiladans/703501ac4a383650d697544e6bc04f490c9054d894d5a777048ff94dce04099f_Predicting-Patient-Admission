# 703501ac4a383650d697544e6bc04f490c9054d894d5a777048ff94dce04099f_Predicting Patient Admission
Predicting Patient Admission

## Setup Instructions (uv installation, creation of virtual environment, and installation dependencies)
1. Install pyenv
If pyenv is not installed, run:
curl https://pyenv.run | bash

Then add the following to your ~/.zshrc (or ~/.bashrc):
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

Reload your terminal:
source ~/.zshrc

2. Install Python 3.12.8
pyenv install 3.12.8
pyenv local 3.12.8

Check that it worked:
python --version  # Should show Python 3.12.8

3. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

4. Install uv (via pipx)
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install uv

5. Initialize the project with uv
uv init

Update in the prompts (project name, version, etc.)

6. Install required dependencies
uv pip install pandas numpy scikit-learn jupyter matplotlib
