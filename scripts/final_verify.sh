#!/usr/bin/env bash
set -euo pipefail

# Final verification for v0.1.0
# Requirements:
# - Use project virtual environment only (.venv)
# - No network calls during tests; offline defaults should already be in env
# - Coverage gate is enforced via pyproject.toml

# Ensure reproducible timestamp is valid for ZIP (>= 1980-01-01)
MIN_EPOCH=315532800
if [[ -z "${SOURCE_DATE_EPOCH:-}" ]] || ! [[ "${SOURCE_DATE_EPOCH}" =~ ^[0-9]+$ ]] || (( SOURCE_DATE_EPOCH < MIN_EPOCH )); then
  export SOURCE_DATE_EPOCH="${MIN_EPOCH}"
fi

# Activate project virtual environment
source .venv/bin/activate

# Run linters/formatters via pre-commit hooks
pre-commit run --all-files

# Type-check
mypy src

# Tests with coverage (XML + terminal); enforce fail_under from pyproject
pytest -q --maxfail=1 --disable-warnings --cov=src --cov-report=term --cov-report=xml

# Ensure 'build' module is available in the venv
if ! python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('build') else 1)"; then
  python -m pip install build
fi

# Build wheel/sdist
python -m build

# Install built wheel locally (no dependency resolution)
python -m pip install --no-deps dist/*.whl

# Check installed package metadata and dependency graph (non-fatal)
python -m pip check || true

echo "Final verification complete"