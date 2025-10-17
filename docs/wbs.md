# Work Breakdown Structure (WBS) — Modular Service Desk Onboarding Analyzer

Purpose
- Translate [docs/architecture.md](docs/architecture.md) and [docs/reference_behavior.md](docs/reference_behavior.md) into sequenced, agent-assigned delivery work.
- Define milestones, deliverables, acceptance criteria, and task dependencies to produce the new app derived from [servicenow_analyzer.py](servicenow_analyzer.py) and [plan.md](plan.md).

Roles (agents)
- API Engineer (FastAPI, Pydantic)
- Engine Engineer (ingest, preprocess, features, analytics)
- AI/LLM Engineer (embeddings, LLM, rerank, prompts)
- Vector/DB Engineer (FAISS/Milvus, SQL/ORM)
- DevOps Engineer (CI, packaging, Docker)
- QA/Testing Engineer (unit, integration, E2E)
- Docs/UX Engineer (docs, Streamlit UI wiring)

Conventions
- Branch per feature; PR requires lint (Ruff/Black/isort), type-check (mypy), tests.
- All modules typed; pure logic in Engine; side-effects in adapters.
- Do not modify [servicenow_analyzer.py](servicenow_analyzer.py).

Milestone M1 — Foundations and Scaffolding
Deliverables
- Repository skeleton: src/, tests/, docs/, config/, .github/workflows/ci.yml
- Tooling: [pyproject.toml](pyproject.toml), [.editorconfig](.editorconfig), [.pre-commit-config.yaml](.pre-commit-config.yaml)
- App entry: [src/api/main.py](src/api/main.py) bootable FastAPI with health endpoint
- Settings: [src/config/settings.py](src/config/settings.py) with env validation
- CI pipeline green on empty app
Acceptance Criteria
- pre-commit passes locally; CI runs lint/type/tests on PR
- GET /health returns status: ok
Tasks
- [API Eng] Create [src/api/main.py](src/api/main.py) with FastAPI app and /health
- [DevOps] Add [pyproject.toml](pyproject.toml) with FastAPI, Uvicorn, Pydantic v2, SQLModel/SQLAlchemy, FAISS-cpu, httpx, numpy, pandas, scikit-learn, sentence-transformers; add dev deps: pytest, pytest-asyncio, mypy, ruff, black, isort
- [DevOps] Configure [.pre-commit-config.yaml](.pre-commit-config.yaml), [.editorconfig](.editorconfig), [./.github/workflows/ci.yml](.github/workflows/ci.yml)
- [Engine Eng] Add [src/config/settings.py](src/config/settings.py) with pydantic-settings; 12-factor envs
- [QA] Add tests for /health and settings validation under [tests/integration/api/](tests/integration/api/)
Dependencies
- None

Milestone M2 — Ingestion and Persistence
Deliverables
- Ingest endpoint [src/api/routers/ingest.py](src/api/routers/ingest.py) handling CSV/Excel
- Normalization and PII cleaning: [src/engine/ingest/loader.py](src/engine/ingest/loader.py), [src/engine/preprocess/clean.py](src/engine/preprocess/clean.py)
- DB models and repositories for datasets/tickets: [src/db/models.py](src/db/models.py), [src/db/repositories/datasets_repo.py](src/db/repositories/datasets_repo.py), [src/db/repositories/tickets_repo.py](src/db/repositories/tickets_repo.py)
- Dev DB: SQLite; migrations stub [src/db/migrations/](src/db/migrations/)
Acceptance Criteria
- POST /ingest/upload stores dataset, tickets with normalized columns and file_hash uniqueness
- GET dataset summary returns row and department counts
Tasks
- [Engine Eng] Implement loader.validate_and_load and preprocess.clean_text/normalize_columns
- [Vector/DB Eng] Implement models and repositories; ensure unique(file_hash)
- [API Eng] Implement ingest router; schema validations; streaming-friendly parsing
- [QA] Tests for invalid schema, duplicate uploads, summary endpoint
Dependencies
- M1 complete

Milestone M3 — Embeddings and Vector Index
Deliverables
- Embedding module: [src/ai/embeddings/openai_embedder.py](src/ai/embeddings/openai_embedder.py)
- Vector store adapter: [src/vector_store/faiss_index.py](src/vector_store/faiss_index.py)
- Endpoint: [src/api/routers/embed.py](src/api/routers/embed.py)
Acceptance Criteria
- POST /embed/run persists embeddings; vector index built and reloadable
- Throughput metrics exposed; errors handled with retries/backoff
Tasks
- [AI Eng] Implement embed_texts with batching and backpressure
- [Vector/DB Eng] Implement FAISS adapter with persist/load
- [API Eng] Implement embed endpoint; idempotent per model/dataset
- [QA] Integration tests covering persistence and reload
Dependencies
- M2 complete

Milestone M4 — Search, Analytics, and Charts
Deliverables
- Search endpoint: [src/api/routers/search.py](src/api/routers/search.py)
- Analytics metrics: [src/engine/analytics/metrics.py](src/engine/analytics/metrics.py)
- Visualizations spec: [src/engine/analytics/visualizations.py](src/engine/analytics/visualizations.py)
- Endpoint: [src/api/routers/analytics.py](src/api/routers/analytics.py)
Acceptance Criteria
- GET /search/nn returns neighbors with scores and filters (department, product)
- GET /analytics/metrics returns distributions mirroring monolith fidelity
Tasks
- [AI Eng] Optional rerank scaffold [src/ai/rerank/cross_encoder.py](src/ai/rerank/cross_encoder.py)
- [Engine Eng] Implement metrics functions; None fallbacks on missing columns
- [API Eng] Implement search and analytics routers; caching where appropriate
- [QA] Tests for filter correctness and performance thresholds
Dependencies
- M3 complete

Milestone M5 — Clustering and Topic Modeling
Deliverables
- Clustering: [src/api/routers/cluster.py](src/api/routers/cluster.py), [src/engine/analytics/clustering.py](src/engine/analytics/clustering.py)
- Persistence: [src/db/repositories/clusters_repo.py](src/db/repositories/clusters_repo.py)
Acceptance Criteria
- silhouette score computed; clusters persisted and retrievable
Tasks
- [Engine Eng] Implement K-Means/HDBSCAN flow and TF-IDF top terms
- [Vector/DB Eng] Repo for cluster assignments and metrics
- [QA] Integration tests with small synthetic dataset
Dependencies
- M4 complete

Milestone M6 — LLM Analysis, Sampling, Reports
Deliverables
- Stratified sampling: [src/engine/features/sampling.py](src/engine/features/sampling.py)
- LLM analysis: [src/ai/llm/analysis.py](src/ai/llm/analysis.py)
- Prompts store: [src/ai/llm/prompts/store.py](src/ai/llm/prompts/store.py) and templates dir
- Reports: [src/api/routers/reports.py](src/api/routers/reports.py), [src/engine/analytics/report.py](src/engine/analytics/report.py)
- History: [src/api/routers/history.py](src/api/routers/history.py), [src/db/repositories/analyses_repo.py](src/db/repositories/analyses_repo.py)
Acceptance Criteria
- POST /analysis/run returns structured markdown; records prompt_version and metrics
- GET /reports/{dataset_id} returns combined markdown with chart counts
- GET /history/analyses paginated; parameterized and safe
Tasks
- [Engine Eng] Implement stratified sampling with caps and summaries
- [AI Eng] Implement analysis pipeline with token budget enforcement
- [Docs/UX] Wire Streamlit UI to APIs in [src/ui/streamlit_app.py](src/ui/streamlit_app.py) (optional)
- [QA] E2E smoke: ingest → embed → search → analysis → report
Dependencies
- M5 complete

Milestone M7 — Observability, Security, and Hardening
Deliverables
- Structured logging and tracing config: [src/config/logging.yaml](src/config/logging.yaml)
- RBAC guards for admin/history routes
- CI gates: coverage threshold, mypy strictness
- Docker packaging; optional Milvus adapter [src/vector_store/milvus_adapter.py](src/vector_store/milvus_adapter.py)
Acceptance Criteria
- Traces for external calls; Prometheus metrics emitted
- RBAC enforced; audit trail for prompt changes
- Reproducible build; tests > 85% coverage for engine and API
Tasks
- [DevOps] Add OpenTelemetry, Prometheus instrumentation
- [API Eng] JWT-based RBAC guards; audit logs
- [Vector/DB Eng] Implement Milvus adapter parity tests (optional)
- [QA] Load tests for 100k tickets profile; backup/restore verification
Dependencies
- M6 complete

Timeline and Sequencing
- Weeks 1-2: M1, M2
- Weeks 3-4: M3, M4
- Weeks 5-6: M5
- Weeks 7-8: M6, M7

Definition of Done (Project)
- All milestones acceptance criteria met and CI green
- Docs updated: [docs/architecture.md](docs/architecture.md), [docs/reference_behavior.md](docs/reference_behavior.md), README
- Tagged release v0.1.0 with changelog

Risk Register (active monitoring)
- Column normalization mismatches; add mapping table tests
- Token budget overruns; enforce truncation and segment summaries
- Vector backend migration complexity; ensure adapter parity and migration tooling
- PII redaction gaps; add automated checks in preprocess tests

References
- [plan.md](plan.md)
- [servicenow_analyzer.py](servicenow_analyzer.py)
- [docs/reference_behavior.md](docs/reference_behavior.md)
- [docs/architecture.md](docs/architecture.md)
