# Service Desk Onboarding Analyzer

Production-grade, modular backend for analyzing onboarding gaps in service desk tickets. Provides ingestion (CSV/Excel), embeddings, offline analysis, and report generation via a FastAPI app and an offline-friendly CLI.

- Offline by default: deterministic builtin embeddings + offline analyzer (no network)
- In-process orchestration: CLI uses FastAPI TestClient; no uvicorn server spin-up
- Contracts-first: CLI calls existing API routers for stable behavior and payloads

## Features

- Ingestion: CSV/XLSX → normalize columns → persist dataset + tickets → summary
- Embeddings: Compute vectors (builtin or sentence-transformers) → persist → build FAISS index
- Analysis: Stratified sampling with token budget → offline analyzer → persist analysis record
- Reports: Assemble markdown with recent analyses and metrics snapshots

## Installation

Use the project virtual environment. Do not use system installs.

```bash
# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode
pip install -e .

# (Optional) Install dev tools for linting/testing
pip install -r requirements-dev.txt  # if present
```

Console script entry point is provided via [pyproject.toml](pyproject.toml).

## Offline Defaults

The CLI and helpers set offline-safe defaults when env vars are missing:

- APP_EMBED_BACKEND=builtin
- ANALYZER_BACKEND=offline
- TRANSFORMERS_OFFLINE=1
- HF_HUB_OFFLINE=1

These prevent network calls and ensure CI determinism. You can override them by setting env vars explicitly.

## CLI

The CLI entry point is `sdonb`. It orchestrates in-process requests to the FastAPI app.

Subcommands:
- ingest: Upload and persist a dataset from a CSV/XLSX
- embed: Compute embeddings and build FAISS index
- analysis: Run offline analysis with token budgeting
- report: Retrieve dataset report markdown
- pipeline: End-to-end (ingest → embed → analysis → report)

### Usage Examples

```bash
# Show help
sdonb --help

# 1) Ingest
sdonb ingest --file data/tickets.csv

# 2) Embeddings (offline builtin backend)
sdonb embed --dataset-id 1 --backend builtin --model-name builtin-384

# 3) Analysis (offline analyzer, prompt v1)
sdonb analysis --dataset-id 1 --question "Top onboarding gaps?" --prompt-version v1 --max-tickets 50 --token-budget 8000

# 4) Report (print to stdout)
sdonb report --dataset-id 1

# 5) Pipeline (end-to-end), print final report markdown
sdonb pipeline --file data/tickets.csv --question "Top onboarding gaps?"

# 6) Pipeline (write markdown to file and print path)
sdonb pipeline --file data/tickets.csv --question "Top onboarding gaps?" --out reports/onboarding.md
```

### Module-based invocation (tests/offline-friendly)

To avoid PATH issues, tests invoke the CLI via module:

```bash
python -m sd_onboarding_analyzer.cli.__main__ pipeline --file data.csv --question "Top onboarding gaps?"
```

## Programmatic API

Helper functions are provided for reuse:

- get_app(): FastAPI app instance with offline defaults
- run_ingest(app, file_path, dataset_name) -> dict
- run_embed(app, dataset_id, backend="builtin", model_name="builtin-384") -> dict
- run_analysis(app, dataset_id, question, prompt_version="v1", max_tickets=50, token_budget=8000) -> dict
- run_report(app, dataset_id) -> dict
- run_pipeline(app, file_path, question, prompt_version="v1") -> dict

These call the app routes via TestClient for offline determinism.

## API Endpoints Summary

- POST /ingest/upload
  - Multipart upload: field "file" (CSV/XLSX)
  - Returns: {dataset_id, name, row_count, department_count, file_hash, inserted_tickets}

- POST /embed/run
  - JSON: {dataset_id, backend: "builtin"|"sentence-transformers", model_name, batch_size?}
  - Returns: {dataset_id, model_name, backend, vector_dim, embedded_count, indexed}

- POST /analysis/run
  - JSON: {dataset_id, question, prompt_version="v1", analyzer_backend="offline", max_tickets=50, token_budget=8000}
  - Returns: {analysis_id, dataset_id, prompt_version, ticket_count, created_at}

- GET /reports/{dataset_id}
  - Returns: {dataset_id, report_markdown, analysis_count}

## Data Model (Canonical Columns)

The ingest pipeline normalizes incoming CSV/XLSX columns to canonical names used across analysis and reporting:

- Department
- Assignment Group
- extract_product
- summarize_ticket
- ticket_quality
- resolution_complexity
- Reassignment group count tracking_index

The engine merges/renames variant input headers into these canonical columns automatically.

## Development

- Code style: Black, isort, Ruff; type-checking with mypy
- Tests: pytest (unit/integration/E2E). Offline integration tests validate ingest → embed → analysis → report without network calls.
- Pre-commit hooks recommended in CI; see CONTRIBUTING.md for details.

Run tests:

```bash
source .venv/bin/activate
pytest -q
```

## Security and Privacy

- PII redaction during normalization for email and phone patterns
- No secrets hardcoded; environment-driven configuration
- Offline-first defaults to avoid unintended external calls

## License

MIT License. See [pyproject.toml](pyproject.toml) for metadata.


## Reranking (Lexical builtin and optional Cross-Encoder)

The semantic search endpoint supports an optional rerank stage on the FAISS top-k candidates.

- Endpoint: `GET /search/nn`
- Query params:
  - `dataset_id`: int (required)
  - `q`: str (required)
  - `k`: int (default 10)
  - `department`: optional repeated filter
  - `product`: optional repeated filter
  - `rerank`: bool (default false)
  - `rerank_backend`: "builtin" | "cross-encoder" (default from env `APP_RERANK_BACKEND`, defaults to "builtin")

Builtin lexical rerank (deterministic, offline):
- Uses regex tokenization, Jaccard + weighted-Jaccard scoring, normalized to [0,1].
- Stable across runs and environments.

Cross-Encoder rerank (optional):
- Requires `sentence-transformers` + `torch` and a locally available Hugging Face model (e.g., "cross-encoder/ms-marco-MiniLM-L6-v2").
- Offline guards: if `TRANSFORMERS_OFFLINE=1` or `HF_HUB_OFFLINE=1`, requests to cross-encoder rerank will fail with a clear error to keep CI deterministic.

Example (HTTPie):
```bash
http GET :8000/search/nn dataset_id==1 q=="login account issue" k==5 rerank==true rerank_backend=="builtin"
```

Example (curl):
```bash
curl -G "http://localhost:8000/search/nn" \
  --data-urlencode "dataset_id=1" \
  --data-urlencode "q=login account issue" \
  --data-urlencode "k=5" \
  --data-urlencode "rerank=true" \
  --data-urlencode "rerank_backend=builtin"
```

Env defaults (CLI/app factory):
- APP_RERANK_BACKEND=builtin (safe offline default)
- To attempt cross-encoder, unset offline flags and ensure model is locally available:
  - TRANSFORMERS_OFFLINE=0
  - HF_HUB_OFFLINE=0

## Docker

Build a reproducible, offline-friendly image using the multi-stage Dockerfile.

Build:
```bash
docker build -t sdonb:local .
```

Run API:
```bash
docker run --rm \
  -e APP_EMBED_BACKEND=builtin \
  -e ANALYZER_BACKEND=offline \
  -e APP_RERANK_BACKEND=builtin \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e APP_FAISS_ENABLED=0 \
  -p 8000:8000 sdonb:local
```

- Default command runs the API: `uvicorn api.main:create_app --host 0.0.0.0 --port 8000`
- The CLI entrypoint `sdonb` is available inside the container.

To enable FAISS native index (optional):
```bash
docker run --rm -e APP_FAISS_ENABLED=1 -p 8000:8000 sdonb:local
```

## Reproducible Packaging

Distributions are reproducible by honoring `SOURCE_DATE_EPOCH`.

- Make targets:
  - `make package` → build wheel/sdist and verify
  - `make release-check` → lint, type-check, test, build, check
- CI/Makefile default `SOURCE_DATE_EPOCH=0` (Unix epoch). In release pipelines, set a stable timestamp:
```bash
export SOURCE_DATE_EPOCH=$(date +%s)   # or a fixed value for releases
make package
```
## Release v0.1.0

Follow the release checklist in [docs/release.md](docs/release.md) for final verification and tagging.

Quick commands (local, .venv only):
```bash
# Activate venv
source .venv/bin/activate

# Run final verification (pre-commit, mypy, pytest+coverage, build, install, pip check)
bash scripts/final_verify.sh

# Tag and push the release
git tag v0.1.0
git push origin v0.1.0

# Build Docker image (offline defaults baked in)
docker build -t sd-onboarding-analyzer:0.1.0 .

# Run API container with offline-safe flags
docker run --rm \
  -e APP_EMBED_BACKEND=builtin \
  -e ANALYZER_BACKEND=offline \
  -e APP_RERANK_BACKEND=builtin \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -p 8000:8000 sd-onboarding-analyzer:0.1.0
```

Notes:
- Offline mode flags and rerank defaults are documented above (see "Offline Defaults" and "Reranking" sections). The system defaults to APP_EMBED_BACKEND=builtin, ANALYZER_BACKEND=offline, and APP_RERANK_BACKEND=builtin for deterministic CI.
- Reproducible builds: honor SOURCE_DATE_EPOCH (see "Reproducible Packaging").
