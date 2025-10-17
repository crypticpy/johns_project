# Contributing Guide

This project implements a production-grade, offline-first pipeline for ingest → embed → analysis → report. Contributions are welcome when aligned with the architecture and quality bar described below.

## Development Environment

- Python 3.10+ (prefer 3.10/3.11/3.12)
- Virtual environment required (no system installs)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional dev tools (if provided by the repo):
```bash
# Example only; install actual dev deps present in your environment
pip install black isort ruff mypy pytest pytest-asyncio
```

## Offline Defaults and Backends

To ensure determinism in CI and local dev:
- APP_EMBED_BACKEND=builtin
- ANALYZER_BACKEND=offline
- TRANSFORMERS_OFFLINE=1
- HF_HUB_OFFLINE=1

The CLI and app factory set these when the env does not specify them. Do not remove these safeguards.

## Workflow

- Branches: feature/*, fix/*, docs/*
- PRs: small, focused, passing all checks
- Commit messages: concise, imperative mood; reference issue IDs when applicable

### Pre-Commit and Linters

Run formatters and linters locally before pushing:
```bash
black src tests
isort src tests
ruff check src tests
mypy src
pytest -q
```

Policies:
- Type hints required for public interfaces
- Keep functions single-responsibility, low coupling
- Avoid network calls in default paths; must be strictly opt-in via env/flags
- No hardcoded secrets; use env/config only

## Testing

- Unit: engine modules, utilities
- Integration: repositories, vector adapters, API routers, CLI (offline)
- E2E: ingest → embed → analysis → report

Run tests:
```bash
source .venv/bin/activate
pytest -q
```

### CLI Integration Tests

Tests should invoke the CLI via module to avoid PATH issues:
```bash
python -m sd_onboarding_analyzer.cli.__main__ pipeline --file data.csv --question "Test"
```

Assertions:
- Exit code 0
- Output includes report header or analytics snapshot or analysis_id

## Style and Quality

- Formatting: Black (line length 100), isort, Ruff
- Type checking: mypy (no implicit Optional, disallow untyped defs)
- Documentation: docstrings for modules/classes/functions using clear, actionable descriptions
- Security: sanitize/validate inputs; no PII leaks; honor offline modes

## Architecture and Boundaries

- Engine: pure logic (no I/O)
- Adapters/Repos: perform I/O (DB, vector stores)
- API: FastAPI routers—contracts via JSON payloads; use defensive parsing (avoid fragile TypeAdapter cases)
- CLI: In-process orchestration using FastAPI TestClient; do not start uvicorn

Do not modify `servicenow_analyzer.py`. Improvements should be implemented in modular engine/services consistent with the architecture.

## Releasing

Follow Keep a Changelog format in CHANGELOG.md. For v0.x releases:
- Ensure offline determinism
- Update README with working examples
- Verify CLI console script entry remains correct