# Makefile for sd-onboarding-analyzer
# Usage:
#   make venv           - Create local virtualenv (.venv)
#   make install        - Install project in editable mode into .venv
#   make lint           - Run pre-commit on all files (if configured)
#   make typecheck      - Run mypy
#   make test           - Run pytest with coverage
#   make build          - Build wheel and sdist (reproducible with SOURCE_DATE_EPOCH)
#   make check-dist     - Run twine check if available; else pip check
#   make package        - Build and check distributions
#   make docker-build   - Build Docker image (offline-friendly defaults)
#   make docker-run     - Run API server in Docker (port 8000)
#   make release-check  - Run release checks script (aggregates lint/type/test/build/check)
#
# Offline-friendly defaults to avoid network in CI:
export APP_EMBED_BACKEND ?= builtin
export ANALYZER_BACKEND ?= offline
export APP_RERANK_BACKEND ?= builtin
export TRANSFORMERS_OFFLINE ?= 1
export HF_HUB_OFFLINE ?= 1
export APP_FAISS_ENABLED ?= 0
# Reproducible builds: when not set by CI, default to 0 (Unix epoch)
export SOURCE_DATE_EPOCH ?= 0

PYTHON := ./.venv/bin/python
PIP := ./.venv/bin/pip
PRE_COMMIT := ./.venv/bin/pre-commit
PYTEST := ./.venv/bin/pytest
MYPY := ./.venv/bin/mypy

DOCKER_IMAGE ?= sdonb:local

.PHONY: venv
venv:
	python3 -m venv .venv

.PHONY: install
install: venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .
	# Optional tools (best-effort)
	-$(PIP) install pre-commit mypy pytest build twine pytest-cov

.PHONY: lint
lint:
	@if [ -x "$(PRE_COMMIT)" ]; then \
		$(PRE_COMMIT) run --all-files ; \
	else \
		echo "pre-commit not installed in .venv; skipping lint"; \
	fi

.PHONY: typecheck
typecheck:
	@if [ -x "$(MYPY)" ]; then \
		$(MYPY) src ; \
	else \
		echo "mypy not installed in .venv; skipping typecheck"; \
	fi

.PHONY: test
test:
	@if [ -x "$(PYTEST)" ]; then \
		$(PYTEST) -q --maxfail=1 --disable-warnings --cov=src --cov-report=term --cov-report=xml ; \
	else \
		echo "pytest not installed in .venv; skipping tests"; \
	fi

.PHONY: build
build:
	$(PYTHON) -m build

.PHONY: check-dist
check-dist:
	@if ./.venv/bin/python -c "import importlib; exit(0 if importlib.util.find_spec('twine') else 1)" ; then \
		./.venv/bin/twine check dist/* ; \
	else \
		echo "twine not installed; running 'pip check' instead"; \
		$(PIP) check ; \
	fi

.PHONY: package
package: build check-dist

.PHONY: docker-build
docker-build:
	docker build -t $(DOCKER_IMAGE) .

.PHONY: docker-run
docker-run:
	docker run --rm -e APP_EMBED_BACKEND=$(APP_EMBED_BACKEND) \
		-e ANALYZER_BACKEND=$(ANALYZER_BACKEND) \
		-e APP_RERANK_BACKEND=$(APP_RERANK_BACKEND) \
		-e TRANSFORMERS_OFFLINE=$(TRANSFORMERS_OFFLINE) \
		-e HF_HUB_OFFLINE=$(HF_HUB_OFFLINE) \
		-e APP_FAISS_ENABLED=$(APP_FAISS_ENABLED) \
		-p 8000:8000 $(DOCKER_IMAGE)

.PHONY: release-check
release-check:
	bash scripts/release_check.sh
