# Multi-stage Dockerfile for sd-onboarding-analyzer
# - Builder: creates project virtualenv, installs package and dev tools, optional test stage
# - Runtime: copies virtualenv and app, sets offline-safe defaults, runs API server
#
# Offline-friendly defaults:
#   APP_EMBED_BACKEND=builtin
#   ANALYZER_BACKEND=offline
#   APP_RERANK_BACKEND=builtin
#   TRANSFORMERS_OFFLINE=1
#   HF_HUB_OFFLINE=1
# To enable FAISS native index, set APP_FAISS_ENABLED=1 at runtime.

############################
# Builder
############################
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (minimal); keep slim and reproducible
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create project virtualenv within workspace to comply with project conventions
RUN python -m venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Install core build tooling first
RUN pip install --upgrade pip setuptools wheel

# Copy only metadata first for better layer caching
COPY pyproject.toml README.md CHANGELOG.md CONTRIBUTING.md ./

# Install project in editable mode + runtime deps
# (Editable allows faster dev iteration; runtime stage will copy the built venv)
RUN pip install -e .

# Install dev/test tooling (builder only)
RUN pip install \
    pytest \
    mypy \
    pre-commit \
    build \
    twine

# Copy source and tests
COPY src ./src
COPY tests ./tests

############################
# Optional test stage
############################
FROM builder AS tester
# Optional: run tests during image build (can be enabled in CI job)
# Respect offline defaults to avoid network calls
ENV APP_EMBED_BACKEND=builtin \
    ANALYZER_BACKEND=offline \
    APP_RERANK_BACKEND=builtin \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    APP_FAISS_ENABLED=0
# Run integration suite quietly; fail if tests fail
RUN pytest -q

############################
# Runtime
############################
FROM python:3.12-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -m -u 10001 appuser

WORKDIR /app

# Copy virtualenv and application code from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/README.md /app/README.md

# Ensure venv is active for all operations
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Offline-safe defaults; override via -e on docker run if needed
ENV APP_EMBED_BACKEND=builtin \
    ANALYZER_BACKEND=offline \
    APP_RERANK_BACKEND=builtin \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    APP_FAISS_ENABLED=0 \
    SOURCE_DATE_EPOCH=0

# Expose API port
EXPOSE 8000

# Switch to non-root
USER appuser

# Default command: run API server
# CLI (sdonb) is also available inside the container
CMD ["uvicorn", "api.main:create_app", "--host", "0.0.0.0", "--port", "8000"]