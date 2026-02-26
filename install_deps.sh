#!/bin/bash
# Use the venv Python (if activated) or fallback to /usr/local/bin/python
PYTHON="${PYTHON:-/usr/local/bin/python}"
# Better: if we're in a venv, use that python
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
fi
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -r requirements.txt
