# Release v0.1.0 — Final Verification and Offline Publishing Guide

This document defines the production-grade, offline-deterministic process to verify and tag the v0.1.0 release. Follow the checklist exactly; do not use system Python or external secrets. Use the project virtual environment (.venv) only.

## 1) Pre-Release Checklist (Local, Offline)

Environment:
- Use `.venv` only: `source .venv/bin/activate`
- Ensure offline defaults are set:
  - `APP_EMBED_BACKEND=builtin`
  - `ANALYZER_BACKEND=offline`
  - `APP_RERANK_BACKEND=builtin`
  - `TRANSFORMERS_OFFLINE=1`
  - `HF_HUB_OFFLINE=1`
  - Disable tracing/OTLP exporters in env/config (runtime observability is off by default)

Run final verification (all gates must pass):
```bash
# Activate venv
source .venv/bin/activate

# Pre-commit (format + lint + type in hooks)
pre-commit run --all-files

# Type-check
mypy src

# Tests + coverage report (gate configured in pyproject coverage settings)
pytest -q --maxfail=1 --disable-warnings --cov=src --cov-report=term --cov-report=xml

# Build distributions
python -m build

# Validate built wheel offline (no deps pulled)
python -m pip install --no-deps dist/*.whl

# Dependency sanity (non-fatal if platform metadata warnings)
python -m pip check || true

echo "Final verification complete"
```

Quick command:
- `scripts/final_verify.sh` — runs all steps above and exits on first failure.

Coverage gate:
- Configured in [pyproject.toml](../pyproject.toml) under `[tool.coverage.report] fail_under = 80`. The suite must meet or exceed this threshold.

## 2) Tagging and GitHub Release (Local)

Version:
- Confirm `[project.version] = "0.1.0"` in [pyproject.toml](../pyproject.toml).

Tag and push:
```bash
git tag v0.1.0
git push origin v0.1.0
```

Release notes:
- Create GitHub Release for `v0.1.0`
- Copy notes from [CHANGELOG.md](../CHANGELOG.md)

Artifacts:
- CI will build wheel/sdist on tag and upload as workflow artifacts (non-publishing).

## 3) Offline Guidance (Runtime Defaults)

To guarantee offline determinism in CI and Docker:
- Set environment variables:
  - `APP_EMBED_BACKEND=builtin`
  - `ANALYZER_BACKEND=offline`
  - `APP_RERANK_BACKEND=builtin`
  - `TRANSFORMERS_OFFLINE=1`
  - `HF_HUB_OFFLINE=1`
- Disable tracing/exporters (keep observability off by default in release builds).

Reproducible builds:
- Honor `SOURCE_DATE_EPOCH` during packaging and Docker builds:
```bash
export SOURCE_DATE_EPOCH=$(date +%s)  # Or a fixed timestamp for reproducibility
python -m build
```

## 4) Docker Build and Run (Offline Defaults)

Build the release image:
```bash
docker build -t sd-onboarding-analyzer:0.1.0 .
```

Run API with offline flags:
```bash
docker run --rm \
  -e APP_EMBED_BACKEND=builtin \
  -e ANALYZER_BACKEND=offline \
  -e APP_RERANK_BACKEND=builtin \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -p 8000:8000 sd-onboarding-analyzer:0.1.0
```

Notes:
- Default command runs the API: `uvicorn api.main:create_app --host 0.0.0.0 --port 8000`
- The CLI entrypoint `sdonb` is available inside the container.
- Optional FAISS native index can be enabled by setting `APP_FAISS_ENABLED=1` at runtime.

## 5) Reference Files

- Changelog: [CHANGELOG.md](../CHANGELOG.md)
- Project metadata: [pyproject.toml](../pyproject.toml)
- Final verification script: [scripts/final_verify.sh](../scripts/final_verify.sh)
- CI workflows:
  - Release (tag-trigger, non-publishing): [.github/workflows/release.yml](../.github/workflows/release.yml)
  - CI (docker-build validation): [.github/workflows/ci.yml](../.github/workflows/ci.yml)

## 6) Summary

Upon completing the checklist and tagging `v0.1.0`, the repository is release-ready with:
- Linters, type-checks, tests, and coverage gate passing
- Wheel/sdist built successfully; local install verified offline
- CI workflows prepared to build and upload artifacts on tag
- Docker image and runtime configured for offline determinism