# Architecture & Design Spec — Modular Service Desk Onboarding Analyzer

Purpose
- Define the production-ready modular architecture derived from the reference monolith [servicenow_analyzer.py](servicenow_analyzer.py) and the program objectives in [plan.md](plan.md).
- Establish clear module boundaries, contracts, data flows, error model, configuration strategy, and repository layout enabling testability, scalability, and safe iteration.

Scope
- Backend service with clean API contracts, optional Streamlit UI consuming APIs, and CLI/batch orchestration.
- Preserve core behaviors documented in [docs/reference_behavior.md](docs/reference_behavior.md) while improving sampling, persistence, and vector intelligence.

Guiding Principles
- High cohesion, low coupling; single-responsibility modules.
- Pure functions for business logic; side-effects contained in adapters.
- Typed Python (3.10+), asyncio for I/O; FastAPI for service endpoints.
- Repositories for persistence; adapters for external services (LLM, embeddings, vector store).
- 12-factor configuration; structured logs and tracing; OWASP-aligned security.

1. Goals and Constraints
- Preserve filtering, chart fidelity, analysis persistence, and report structure as per monolith references in [servicenow_analyzer.py](servicenow_analyzer.py).
- Improve:
  - Stratified sampling (Department × Topic × Quality) with token budget adherence.
  - Full dataset/ticket persistence, prompt versioning, embeddings, retrieval, rerank, clustering.
  - Observability, RBAC, PII handling, performance targets.
- Constraints:
  - Dev storage: SQLite + FAISS; Prod: Postgres + Milvus/Weaviate.
  - Token budget: prompt size < 16k; 95th percentile latency targets for standard queries.
  - Documentation-first; avoid modifying [servicenow_analyzer.py](servicenow_analyzer.py).

2. Architecture Overview (Layered)
- UI
  - Streamlit initial UI consuming APIs; later React optional. Mirrors [main()](servicenow_analyzer.py:392) tab flows through service endpoints.
- API (FastAPI)
  - Routers: ingest, embed, search, cluster, analytics, suggest, analysis, reports, admin, history.
  - Contracts via Pydantic models; async handlers; consistent error taxonomy.
- Engine
  - Ingest, preprocess, features (sampling), analytics (metrics/visualizations), report assembly.
  - Pure logic; no network or DB I/O.
- AI
  - Embeddings (OpenAI/SentenceTransformers), LLM analysis, rerank (cross-encoder or LLM-based).
- Vector Store
  - Adapters for FAISS (dev) and Milvus/Weaviate (prod); index durability and recovery.
- DB/Repositories
  - Models and repositories for datasets, tickets, embeddings, clusters, analyses, prompts, runs.
- Services
  - Batch orchestration pipelines; backpressure, retries, and monitoring.
- Config/Prompts
  - Centralized settings; prompt templates with versioning and metadata.

3. Module Map and Contracts
- api/routers/ingest.py
  - POST /ingest/upload: Accept Excel/CSV, validate, normalize, persist dataset + tickets; return dataset summary.
  - Dependencies: engine.ingest.loader, db.repositories.datasets_repo, tickets_repo.
- api/routers/embed.py
  - POST /embed/run: Generate embeddings for dataset/tickets; persist; update vector index.
  - Dependencies: ai.embeddings, vector_store.adapter, db.repositories.embeddings_repo.
- api/routers/search.py
  - GET /search/nn: kNN search by text or ticket_id with filters; optional rerank; return neighbors+scores.
  - Dependencies: vector_store.adapter, ai.rerank (optional), repositories.
- api/routers/cluster.py
  - POST /cluster/run: Cluster embeddings (K-Means/HDBSCAN), compute metrics (silhouette), persist clusters.
  - Dependencies: engine.analytics.clustering, embeddings_repo, clusters_repo.
- api/routers/analytics.py
  - GET /analytics/metrics: Return dataset metrics (quality, complexity, department volume, reassignment, product).
  - Dependencies: engine.analytics.metrics, repositories; cache-able results.
- api/routers/suggest.py
  - POST /suggest/kv: Retrieve → rerank → summarize; output KV draft with structured fields; link prompt_version.
  - Dependencies: ai.llm.analysis, ai.rerank, prompts_repo, analyses_repo.
- api/routers/analysis.py
  - POST /analysis/run: Build stratified context; execute LLM analysis; persist analysis record with metrics and prompt_version.
  - Dependencies: engine.features.sampling, ai.llm.analysis, analyses_repo.
- api/routers/reports.py
  - GET /reports/{dataset_id}: Assemble markdown report referencing analyses and chart metrics.
  - Dependencies: engine.analytics.report, analyses_repo, metrics module.
- api/routers/admin.py
  - POST /admin/prompts: Manage prompt templates, versioning, metadata; rollback/version pinning.
  - Dependencies: prompts_repo.
- api/routers/history.py
  - GET /history/analyses: Paginated recent analyses; filter by dataset_id, prompt_version, timestamp range.
  - Dependencies: analyses_repo.

- engine/ingest/loader.py
  - validate_and_load(file) -> LoadedDataset: parse Excel/CSV; compute file_hash; return normalized DataFrame and metadata.
- engine/preprocess/clean.py
  - clean_text, normalize columns (canonical schema), detect/redact PII.
- engine/features/sampling.py
  - stratified_sample(df, config) -> SampleSegments: bucket by Department × Topic × Quality with caps and segment summaries.
- engine/analytics/metrics.py
  - compute_quality_distribution(df), compute_complexity_distribution(df), compute_department_volume(df), compute_reassignment_distribution(df), compute_product_distribution(df).
- engine/analytics/visualizations.py
  - chart_* functions returning Figure specs from metrics payloads (UI renders; API returns data).
- engine/analytics/report.py
  - assemble_markdown(dataset, analyses, metrics_summary) -> str.

- ai/embeddings/openai_embedder.py
  - embed_texts(items, model) -> List[Embedding]; batch with backpressure; metadata capture.
- ai/rerank/cross_encoder.py
  - rerank(candidates, query) -> ranked list; pluggable scoring strategies.
- ai/llm/analysis.py
  - run_analysis(context, question, prompt_version, comparison_mode) -> AnalysisResult; structured content; error-safe.
- ai/llm/prompts/store.py
  - load_template(version), list_versions(), save_template(version, metadata).
  - Templates in ai/llm/prompts/templates/* with versioned filenames.

- vector_store/faiss_index.py
  - build_index(embeddings), add(embeddings), search(query_vector, k, filters) -> neighbors; persist/load index.
- vector_store/weaviate_adapter.py, vector_store/milvus_adapter.py
  - Unified interface parity with FAISS adapter for production deployments.

- db/models.py
  - Pydantic/SQLModel schemas for datasets, tickets, embeddings, clusters, analyses, prompts, runs; consistent field names.
- db/repositories/*
  - datasets_repo: create/get/list; unique by file_hash.
  - tickets_repo: bulk_insert; query with filters.
  - embeddings_repo: persist per ticket; retrieve for clustering/search.
  - clusters_repo: store cluster assignments and metrics.
  - analyses_repo: save analysis records; retrieve history; JSON metrics field; prompt_version linkage.
  - prompts_repo: CRUD for templates + metadata.
  - runs_repo: record batch/service runs for observability.

- services/batch_runner.py
  - Orchestrate ingest → embed → index → analysis; retries/backoff; writes to runs_repo; metrics/tracing.

- config/settings.py
  - AppSettings: database URL, vector backend type, embedding model, Azure/OpenAI credentials, logging/tracing config; environment-loaded and validated.

- utils/*
  - helpers for ID generation, time handling, validation; no business logic.

4. Data Flow
- Ingest: Upload → validate/normalize → persist dataset/tickets → compute summary.
- Embeddings: Generate for tickets → persist → build/update vector index.
- Search: Query text → embed → kNN → optional rerank → return neighbors; filters applied.
- Analytics: Compute distributions/metrics → API returns structured payloads; UI renders charts.
- Clustering: Run algorithm (K-Means/HDBSCAN) → compute silhouette → persist clusters.
- Suggestion: Retrieve nearest tickets → rerank → summarize to KV draft → persist analysis with prompt_version.
- Analysis: Stratified context → LLM analysis → metrics + markdown persisted.
- Reports: Assemble markdown referencing dataset/analyses/metrics; downloadable artifact.

5. Error Model and Handling
- BaseError: AppError(msg, code, details)
  - ValidationError: Bad input/schema violations.
  - NotFoundError: Missing dataset/ticket/analysis.
  - ExternalServiceError: LLM/embedding/vector backend failures; include provider-specific codes.
  - PersistenceError: DB transaction failures; retry or fail-fast with precise message.
  - ConfigError: Missing/invalid configuration; fail at startup with guardrails.
  - RateLimitError: Provider throttling; exponential backoff; user-facing 429 with retry-after.
- Patterns:
  - Guard clauses in API; map exceptions to HTTP status codes with structured JSON responses.
  - Engine remains exception-free by contract (inputs validated pre-call).
  - Services implement retries/backoff/circuit breakers for external calls.

6. Configuration Strategy
- 12-factor env-first configuration via [config/settings.py](config/settings.py).
- Secrets via environment variables (no hardcoding); support Azure/OpenAI and vector backends.
- Logging/tracing config in [config/logging.yaml](config/logging.yaml); OpenTelemetry exporter configurable.
- Prompt templates: filesystem + DB metadata tracked; version pinning by analysis run.

7. Logging, Metrics, and Tracing
- JSON structured logs with contextual fields (dataset_id, run_id, endpoint, latency).
- Prometheus metrics:
  - request_latency_seconds by endpoint
  - embeddings_throughput
  - vector_search_latency
  - analysis_token_usage
- OpenTelemetry tracing:
  - Span per request; child spans for external calls (LLM, embeddings, vector store, DB).
- Runs table:
  - services/runs record start/end, status, counts, error messages for operability.

8. Security and Governance
- Input validation for all payloads; rigid schema enforcement.
- PII detection/redaction in preprocess; prevent raw sensitive fields in LLM prompts.
- RBAC (role claims in JWT) guarding admin/history endpoints.
- Audit logs of admin/prompt changes; retention policy configured.
- Prompt injection defenses: sanitize inputs; template constraints; avoid user-controlled system prompts.

9. Performance and Caching
- Batch embeddings with bounded concurrency and backpressure; queue instrumentation.
- Cache analytics metrics keyed by dataset_id + filter params; TTL-based invalidation.
- Avoid N+1 queries via repository bulk operations; index critical columns.
- Vector index durability: checkpointing; lazy-load on startup; background rebuild if necessary.
- Latency targets:
  - /analytics/metrics 95th < 500ms on dev hardware for 100k tickets.
  - /search/nn 95th < 800ms with FAISS + simple filters; rerank adds bounded overhead.

10. Testability and CI
- Unit tests (pytest) for engine modules; property-based tests for sampling and metrics correctness.
- Integration tests for repositories and vector adapters; local FAISS index operations.
- API tests (FastAPI TestClient) for contracts and error mapping.
- E2E smoke: ingest → embed → search → analysis → report, asserting invariants from [docs/reference_behavior.md](docs/reference_behavior.md).
- Coverage measured; gate in CI; pre-commit hooks (Black, isort, Ruff, mypy).

11. Repository Layout
- src/
  - api/
    - main.py
    - routers/
      - ingest.py
      - embed.py
      - search.py
      - cluster.py
      - analytics.py
      - suggest.py
      - analysis.py
      - reports.py
      - admin.py
      - history.py
  - engine/
    - ingest/loader.py
    - preprocess/clean.py
    - features/sampling.py
    - analytics/metrics.py
    - analytics/visualizations.py
    - analytics/report.py
  - ai/
    - embeddings/openai_embedder.py
    - rerank/cross_encoder.py
    - llm/analysis.py
    - llm/prompts/store.py
    - llm/prompts/templates/ (versioned files)
  - vector_store/
    - faiss_index.py
    - milvus_adapter.py
    - weaviate_adapter.py
  - db/
    - models.py
    - repositories/
      - datasets_repo.py
      - tickets_repo.py
      - embeddings_repo.py
      - clusters_repo.py
      - analyses_repo.py
      - prompts_repo.py
      - runs_repo.py
    - migrations/
  - services/
    - batch_runner.py
  - config/
    - settings.py
    - logging.yaml
  - utils/
    - ids.py
    - time.py
    - validation.py
  - cli/
    - __main__.py
    - run_pipeline.py
  - ui/
    - streamlit_app.py
- tests/
  - unit/engine/*
  - integration/api/*
  - integration/db/*
  - e2e/smoke/*
- docs/
  - architecture.md
  - reference_behavior.md
  - README.md, CONTRIBUTING.md, CHANGELOG.md
- pyproject.toml, .editorconfig, .pre-commit-config.yaml, Dockerfile, .github/workflows/ci.yml

12. Contracts and Schemas (Summary)
- Dataset
  - id, name, file_hash (unique), row_count, department_count, upload_time, metadata (JSON).
- Ticket
  - id, dataset_id FK, department, assignment_group, product, summary fields, quality, complexity, reassignment_count, normalized text.
- Embedding
  - id, ticket_id FK, model_name, vector, created_at.
- Cluster
  - id, dataset_id FK, algorithm, params, assignments (ticket_id → cluster_id), metrics (silhouette, top_terms).
- Analysis
  - id, dataset_id FK, prompt_version, question, result_markdown, metrics JSON, ticket_count, filters, timestamp.
- Prompt
  - version, name, template_path, metadata (author, date, notes), is_active.
- Run
  - id, pipeline_name, status, counts, started_at, finished_at, error_message.

13. Acceptance Criteria Traceability
- M1 Foundations:
  - Upload/persist dataset/tickets; dataset summary endpoint returns counts.
- M2 Embeddings + Search:
  - Embeddings persisted; vector index durable; /search/nn returns top-k with filters.
- M3 Analytics:
  - Metrics endpoints produce distributions; UI charts fidelity; cached responses.
- M4 Clustering:
  - Clusters persisted; silhouette reported; API to retrieve assignments and metrics.
- M5 KV Suggestion:
  - Structured KV drafts; prompt_version recorded; ≥70% human-review usefulness target (tracked via feedback field in analyses).
- M6 Sampling + Comparison:
  - Stratified sampling; comparison mode supported in analysis; token budgets enforced.
- M7 Hardening:
  - CI gates; observability; RBAC; PII; backup/restore; optional Milvus adapter validated.

14. Implementation Notes and Guardrails
- Do not import or modify [servicenow_analyzer.py](servicenow_analyzer.py); use it as behavioral reference only.
- Repository interfaces must be async where I/O-bound; engine functions remain pure/sync unless justified.
- All API inputs validated; pagination and filters for history/analytics endpoints.
- Default dev stack baked into config (SQLite + FAISS) to reduce boot friction; prod flags enable Postgres + Milvus/Weaviate.
- Maintain prompt governance; analyses must link to prompt_version and template metadata.

15. Risks and Open Questions
- Canonical schema mappings for ServiceNow exports (column names normalization) — confirm with ingestion design in engine.
- Rerank model choice (cross-encoder vs LLM) and latency/cost trade-offs — document defaults and allow configuration.
- PII policies and audit scope — finalize in security governance.
- Vector backend migration (FAISS → Milvus/Weaviate) compatibility — define index schema parity and migration tooling.
- Token budgets enforcement strategy — segment summaries, truncation rules, and telemetry fields.
- SQL safety for history limits and filtering — parameterize queries; validate integers; add indices.

16. References
- Behavioral reference: [servicenow_analyzer.py](servicenow_analyzer.py)
- Plan: [plan.md](plan.md)
- Behavior doc: [docs/reference_behavior.md](docs/reference_behavior.md)

End of Architecture & Design Spec.