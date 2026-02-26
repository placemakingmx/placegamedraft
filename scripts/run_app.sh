#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
PORT=${PORT:-8501}

if [ ! -d "$VENV_DIR" ]; then
  echo "Virtualenv not found at $VENV_DIR. Run scripts/create_venv.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Kill any running streamlit processes (best-effort)
pkill -f streamlit || true

# Run Streamlit, bind to 0.0.0.0 so it is accessible from host
streamlit run streamlit_app.py --server.port $PORT --server.headless true --server.enableCORS false
