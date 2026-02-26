#!/usr/bin/env bash
set -euo pipefail

# Creates a virtual environment in .venv and installs requirements
PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

echo "Using Python: $(which $PYTHON)"
$PYTHON -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  echo "requirements.txt not found; please create it or install dependencies manually."
fi

echo "Virtualenv created at $VENV_DIR and dependencies installed. Activate with: source $VENV_DIR/bin/activate"
