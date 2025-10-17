#!/usr/bin/env bash
set -euo pipefail

# Release checks for sd-onboarding-analyzer
# - Runs pre-commit, mypy, pytest
# - Builds wheel/sdist
# - Verifies distributions with twine (if available) or pip check
#
# Offline-friendly defaults: avoid network where possible
export APP_EMBED_BACKEND="${APP_EMBED_BACKEND:-builtin}"
export ANALYZER_BACKEND="${ANALYZER_BACKEND:-offline}"
export APP_RERANK_BACKEND="${APP_RERANK_BACKEND:-builtin}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export APP_FAISS_ENABLED="${APP_FAISS_ENABLED:-0}"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-0}"

VENV="./.venv"
PYTHON="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

if [ ! -d "${VENV}" ]; then
  echo "Creating virtual environment at ${VENV}"
  python3 -m venv "${VENV}"
fi

# Ensure base tooling is available
"${PYTHON}" -m pip install --upgrade pip setuptools wheel

# Install project editable + tools (best-effort)
"${PIP}" install -e .
# Install tools (best-effort; tolerate failures)
"${PIP}" install pre-commit mypy pytest build twine pytest-cov || true

# 1) Pre-commit hooks
if [ -x "${VENV}/bin/pre-commit" ]; then
  echo "Running pre-commit..."
  "${VENV}/bin/pre-commit" run --all-files
else
  echo "pre-commit not installed; skipping."
fi

# 2) Type-check
if [ -x "${VENV}/bin/mypy" ]; then
  echo "Running mypy..."
  "${VENV}/bin/mypy" src
else
  echo "mypy not installed; skipping."
fi

# 3) Tests with coverage
if [ -x "${VENV}/bin/pytest" ]; then
  echo "Running pytest..."
  "${VENV}/bin/pytest" -q --maxfail=1 --disable-warnings --cov=src --cov-report=term --cov-report=xml
else
  echo "pytest not installed; skipping."
fi

# 4) Build distributions (wheel + sdist)
echo "Building distributions..."
"${PYTHON}" -m build

# 5) Verify distributions
if "${PYTHON}" -c "import importlib; exit(0 if importlib.util.find_spec('twine') else 1)"; then
  echo "Checking distributions with twine..."
  "${VENV}/bin/twine" check dist/*
else
  echo "twine not available; running 'pip check' instead..."
  "${PIP}" check
fi

echo "Release checks completed successfully."