# Tools Registry — Design Specification (Function-Calling Backplane)

Overview

Purpose
- Define the formal Tools Registry design to back safe function-calling inside [src/ai/llm/openai_analyzer.py](src/ai/llm/openai_analyzer.py).
- Align with implemented FastAPI routers and module boundaries to ensure schema-consistent, RBAC- and audit-aware execution.
- Non-goals: No code changes. This document is a normative spec and mapping reference only.

Scope confirmation
- App entry: [src/api/main.py](src/api/main.py)
- Routers: [src/api/routers/ingest.py](src/api/routers/ingest.py), [src/api/routers/embed.py](src/api/routers/embed.py), [src/api/routers/search.py](src/api/routers/search.py), [src/api/routers/cluster.py](src/api/routers/cluster.py), [src/api/routers/analysis.py](src/api/routers/analysis.py), [src/api/routers/reports.py](src/api/routers/reports.py), [src/api/routers/history.py](src/api/routers/history.py), [src/api/routers/metrics.py](src/api/routers/metrics.py)
- Security/Audit: [src/security/auth.py](src/security/auth.py), [src/db/repositories/audit_repo.py](src/db/repositories/audit_repo.py), [src/db/models.py](src/db/models.py)
- Observability: [src/observability/metrics.py](src/observability/metrics.py), [src/observability/tracing.py](src/observability/tracing.py)
- Analyzer adapters: [src/ai/llm/openai_analyzer.py](src/ai/llm/openai_analyzer.py), [src/ai/llm/offline_analyzer.py](src/ai/llm/offline_analyzer.py), [src/ai/llm/factory.py](src/ai/llm/factory.py)
- Tools catalog source: [docs/tools_plan.md](docs/tools_plan.md)

--------------------------------------------------------------------------------

Authoritative Tool Mapping

Notes
- Names follow tools_plan convention: tool.<domain>.<action>
- Router alignment verified; HTTP method and path taken from code; request/response shape aligned to actual handlers.

1) tool.ingest.upload
- Router: POST /ingest/upload
- Path reference: [router.post("/upload")](src/api/routers/ingest.py:18)
- Request model: Multipart form with file field (adapter converts file_path → multipart)
- Response model: { dataset_id, name, row_count, department_count, file_hash, inserted_tickets }
- RBAC roles: None enforced by router; Registry: none required
- Audit: None
- Idempotency: create-or-get by file_hash; avoids duplicate ticket insert if dataset exists

2) tool.embed.run
- Router: POST /embed/run
- Path reference: [router.post("/run")](src/api/routers/embed.py:19)
- Request model: { dataset_id, model_name?, batch_size?, backend? }
- Response model: { dataset_id, model_name, backend, vector_dim, embedded_count, indexed }
- RBAC roles: None enforced by router; Registry: none required
- Audit: None
- Idempotency: upsert per (dataset_id, model_name, ticket_id); index rebuilt from DB

3) tool.search.nn
- Router: GET /search/nn
- Path reference: [router.get("/nn")](src/api/routers/search.py:34)
- Request model: query params { dataset_id, q, k?, department[]?, product[]?, rerank?, rerank_backend? }
- Response model: { dataset_id, k, backend, model_name, rerank, rerank_backend, results: [ { ticket_id, score, department?, product?, summary? } ] }
- RBAC roles: None enforced by router; Registry: none required
- Audit: None
- Idempotency: read-only

4) tool.cluster.run
- Router: POST /cluster/run
- Path reference: [router.post("/run")](src/api/routers/cluster.py:46)
- Request model: { dataset_id, algorithm: "kmeans"|"hdbscan", params: { ... }, model_name? }
- Response model: { dataset_id, algorithm, model_name, run_id, silhouette|null, cluster_counts: { cluster_id: count } }
- RBAC roles: Router: none; Registry: {"analyst","admin"} per tools_plan
- Audit: Recommended (tools_plan); Router currently none
- Idempotency: distinct runs; safe repeat given same inputs

5) tool.analysis.run
- Router: POST /analysis/run
- Path reference: [router.post("/run")](src/api/routers/analysis.py:128)
- Request model: { dataset_id, question, prompt_version?, analyzer_backend?, max_tickets?, token_budget?, compare_dataset_id? }
- Response model: { analysis_id, dataset_id, prompt_version, ticket_count, created_at }
- RBAC roles: Router: {"analyst","admin"} when APP_ENABLE_RBAC=true via [require_roles()](src/security/auth.py:67)
- Audit: Router emits via [AuditRepository.record()](src/db/repositories/audit_repo.py:15)
- Idempotency: not idempotent; creates new record per run

6) tool.reports.get
- Router: GET /reports/{dataset_id}
- Path reference: [router.get("/{dataset_id}")](src/api/routers/reports.py:107)
- Request model: { dataset_id } (path)
- Response model: { dataset_id, report_markdown, analysis_count }
- RBAC roles: None enforced by router; Registry: none required
- Audit: None
- Idempotency: read-only

7) tool.prompts.list / tool.prompts.load / tool.prompts.save
- Adapter: Prompt store module (not a router)
- Module reference: [store.py](src/ai/llm/prompts/store.py)
- list: {} → { versions: [string] }
- load: { version } → { version, template, metadata }
- save: { version, template, metadata } → { ok }
- RBAC roles: save → {"admin"} per tools_plan; list/load → none
- Audit: save recommended; module-level hook via Registry
- Idempotency: list/load read-only; save upsert

8) tool.history.list
- Router: GET /history/analyses
- Path reference: [router.get("/analyses")](src/api/routers/history.py:30)
- Request model: { limit, offset, dataset_id?, prompt_version?, date_from?, date_to? } as query
- Response model: { limit, offset, total, items: [ { id, dataset_id, prompt_version, question, ticket_count, created_at } ] }
- RBAC roles: Router: {"viewer","admin"} via [require_roles()](src/security/auth.py:67)
- Audit: None
- Idempotency: read-only

--------------------------------------------------------------------------------

Normative JSON Schemas (Draft 2020-12)

Conventions
- $id: "tool.<name>.input" / "tool.<name>.output"
- additionalProperties: false
- Nullable fields represented via oneOf with type "null" where applicable
- All numbers use type "integer" or "number" as appropriate; constraints stated

tool.ingest.upload.input
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.ingest.upload.input",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "file_path": { "type": "string", "minLength": 1 },
    "dataset_name": { "type": "string", "minLength": 1 }
  },
  "required": ["file_path"]
}

tool.ingest.upload.output
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.ingest.upload.output",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "dataset_id": { "type": "integer", "minimum": 1 },
    "name": { "type": "string" },
    "row_count": { "type": "integer", "minimum": 0 },
    "department_count": { "type": "integer", "minimum": 0 },
    "file_hash": { "type": "string" },
    "inserted_tickets": { "type": "integer", "minimum": 0 }
  },
  "required": ["dataset_id", "name", "row_count", "department_count", "file_hash", "inserted_tickets"]
}

tool.embed.run.input
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.embed.run.input",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "dataset_id": { "type": "integer", "minimum": 1 },
    "backend": { "type": "string", "enum": ["sentence-transformers", "builtin"] },
    "model_name": { "type": "string", "minLength": 1 },
    "batch_size": { "type": "integer", "minimum": 1 }
  },
  "required": ["dataset_id"],
  "allOf": [
    { "if": { "required": ["backend"], "properties": { "backend": { "const": "sentence-transformers" } } },
      "then": { "required": ["model_name"] } }
  ]
}

tool.embed.run.output
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.embed.run.output",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "dataset_id": { "type": "integer", "minimum": 1 },
    "model_name": { "type": "string" },
    "backend": { "type": "string" },
    "vector_dim": { "type": "integer", "minimum": 0 },
    "embedded_count": { "type": "integer", "minimum": 0 },
    "indexed": { "type": "boolean" }
  },
  "required": ["dataset_id", "model_name", "backend", "vector_dim", "embedded_count", "indexed"]
}

tool.search.nn.input
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.search.nn.input",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "dataset_id": { "type": "integer", "minimum": 1 },
    "query_text": { "type": "string", "minLength": 1 },
    "k": { "type": "integer", "minimum": 1 },
    "filters": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "department": { "type": "array", "items": { "type": "string" } },
        "product": { "type": "array", "items": { "type": "string" } }
      }
    },
    "rerank": { "type": "boolean" },
    "rerank_backend": { "type": "string", "enum": ["builtin", "cross-encoder"] }
  },
  "required": ["dataset_id", "query_text"]
}

tool.search.nn.output
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.search.nn.output",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "dataset_id": { "type": "integer", "minimum": 1 },
    "k": { "type": "integer", "minimum": 1 },
    "backend": { "type": ["string", "null"] },
    "model_name": { "type": "string" },
    "rerank": { "type": "boolean" },
    "rerank_backend": { "type": ["string", "null"], "enum": ["builtin", "cross-encoder", null] },
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "ticket_id": { "type": "integer", "minimum": 1 },
          "score": { "type": "number" },
          "department": { "type": ["string", "null"] },
          "product": { "type": ["string", "null"] },
          "summary": { "type": ["string", "null"] }
        },
        "required": ["ticket_id", "score"]
      }
    }
  },
  "required": ["dataset_id", "k", "model_name", "results"]
}

tool.cluster.run.input
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.cluster.run.input",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "dataset_id": { "type": "integer", "minimum": 1 },
    "algorithm": { "type": "string", "enum": ["kmeans", "hdbscan"] },
    "params": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "n_clusters": { "type": "integer", "minimum": 2 },
        "min_cluster_size": { "type": "integer", "minimum": 2 },
        "min_samples": { "type": "integer", "minimum": 1 }
      }
    },
    "model_name": { "type": "string" }
  },
  "required": ["dataset_id", "algorithm"]
}

tool.cluster.run.output
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.cluster.run.output",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "dataset_id": { "type": "integer", "minimum": 1 },
    "algorithm": { "type": "string", "enum": ["kmeans", "hdbscan"] },
    "model_name": { "type": "string" },
    "run_id": { "type": "integer", "minimum": 1 },
    "silhouette": { "type": ["number", "null"] },
    "cluster_counts": {
      "type": "object",
      "additionalProperties": { "type": "integer", "minimum": 0 }
    }
  },
  "required": ["dataset_id", "algorithm", "model_name", "run_id", "cluster_counts"]
}

tool.analysis.run.input
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.analysis.run.input",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "dataset_id": { "type": "integer", "minimum": 1 },
    "question": { "type": "string", "minLength": 1 },
    "prompt_version": { "type": "string", "minLength": 1, "default": "v1" },
    "analyzer_backend": { "type": "string", "enum": ["openai", "offline"] },
    "max_tickets": { "type": "integer", "minimum": 1, "default": 50 },
    "token_budget": { "type": "integer", "minimum": 1, "default": 2000 },
    "compare_dataset_id": { "type": "integer", "minimum": 1 }
  },
  "required": ["dataset_id", "question"]
}

tool.analysis.run.output
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.analysis.run.output",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "analysis_id": { "type": "integer", "minimum": 1 },
    "dataset_id": { "type": "integer", "minimum": 1 },
    "prompt_version": { "type": "string" },
    "ticket_count": { "type": "integer", "minimum": 0 },
    "created_at": { "type": ["string", "null"], "format": "date-time" }
  },
  "required": ["analysis_id", "dataset_id", "prompt_version", "ticket_count"]
}

tool.reports.get.input
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.reports.get.input",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "dataset_id": { "type": "integer", "minimum": 1 }
  },
  "required": ["dataset_id"]
}

tool.reports.get.output
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.reports.get.output",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "dataset_id": { "type": "integer", "minimum": 1 },
    "report_markdown": { "type": "string" },
    "analysis_count": { "type": "integer", "minimum": 0 }
  },
  "required": ["dataset_id", "report_markdown", "analysis_count"]
}

tool.prompts.list.input
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.prompts.list.input",
  "type": "object",
  "additionalProperties": false,
  "properties": {},
  "required": []
}

tool.prompts.list.output
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.prompts.list.output",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "versions": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["versions"]
}

tool.prompts.load.input
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.prompts.load.input",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "version": { "type": "string", "minLength": 1 }
  },
  "required": ["version"]
}

tool.prompts.load.output
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.prompts.load.output",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "version": { "type": "string" },
    "template": { "type": "string" },
    "metadata": { "type": "object" }
  },
  "required": ["version", "template"]
}

tool.prompts.save.input
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.prompts.save.input",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "version": { "type": "string", "minLength": 1 },
    "template": { "type": "string", "minLength": 1 },
    "metadata": { "type": "object" }
  },
  "required": ["version", "template"]
}

tool.prompts.save.output
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.prompts.save.output",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "ok": { "type": "boolean" }
  },
  "required": ["ok"]
}

tool.history.list.input
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.history.list.input",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "limit": { "type": "integer", "minimum": 1, "maximum": 500, "default": 50 },
    "offset": { "type": "integer", "minimum": 0, "default": 0 },
    "dataset_id": { "type": "integer", "minimum": 1 },
    "prompt_version": { "type": "string" },
    "date_from": { "type": "string", "format": "date-time" },
    "date_to": { "type": "string", "format": "date-time" }
  },
  "required": ["limit", "offset"]
}

tool.history.list.output
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "tool.history.list.output",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "limit": { "type": "integer", "minimum": 1 },
    "offset": { "type": "integer", "minimum": 0 },
    "total": { "type": "integer", "minimum": 0 },
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "id": { "type": "integer", "minimum": 1 },
          "dataset_id": { "type": "integer", "minimum": 1 },
          "prompt_version": { "type": "string" },
          "question": { "type": "string" },
          "ticket_count": { "type": "integer", "minimum": 0 },
          "created_at": { "type": ["string", "null"], "format": "date-time" }
        },
        "required": ["id", "dataset_id", "prompt_version", "question", "ticket_count"]
      }
    }
  },
  "required": ["limit", "offset", "total", "items"]
}

--------------------------------------------------------------------------------

Tool Registry Interface (Responsibilities)

Registration rules
- Stable tool names exactly match the catalog in [docs/tools_plan.md](docs/tools_plan.md).
- Each tool registers:
  - name (string), description
  - input_schema ($id tool.<name>.input), output_schema ($id tool.<name>.output)
  - adapter function binding
  - RBAC role set (may differ from router enforcement; registry enforces proactively)

Lookup and validation pipeline
- On tool call:
  - Validate name against registry set; non-whitelisted names refused.
  - Validate args against tool.<name>.input using strict JSON Schema (additionalProperties=false).
  - Perform RBAC check via [require_roles()](src/security/auth.py:67) when roles are specified for the tool.
  - Execute adapter; collect result.
  - Validate result against tool.<name>.output; if mismatch, map to validation_error.

RBAC check integration points
- Registry runs RBAC checks regardless of router enforcement to ensure early denial.
- Router-level RBAC remains authoritative where implemented:
  - Analysis: [require_roles({"analyst","admin"})](src/api/routers/analysis.py:155)
  - History: [require_roles({"viewer","admin"})](src/api/routers/history.py:34)
- For tools without router RBAC (e.g., cluster.run), registry enforces {"analyst","admin"} as per tools_plan.

Audit emission hook
- Emit audit on successful completion for sensitive tools:
  - analysis.run (already in router via [AuditRepository.record()](src/db/repositories/audit_repo.py:15))
  - cluster.run, prompts.save (registry layer emits using [AuditRepository.record()](src/db/repositories/audit_repo.py:15))
- Audit record fields:
  - subject (JWT sub or "anonymous")
  - action (tool name, e.g., "analysis.run")
  - resource ("dataset:{id}" or "prompts:{version}")
  - metadata (pii-free): ids, prompt_version, parameters summary

Metrics emission fields
- Increment per-call counters and measure latency:
  - tool_call_count{tool_name}
  - tool_latency_seconds{tool_name}
- Optional domain metrics reuse [EMBEDDINGS_THROUGHPUT](src/observability/metrics.py:46), [VECTOR_SEARCH_LATENCY](src/observability/metrics.py:51), [ANALYSIS_TOKEN_USAGE](src/observability/metrics.py:56).
- Request-level metrics already emitted via [MetricsMiddleware](src/observability/metrics.py:84) and [metrics endpoint](src/api/routers/metrics.py:11).

Tracing span attributes
- When tracing enabled:
  - Start span "tool.call" with attributes:
    - tool_name
    - args_summary (sanitized; counts and identifiers only)
    - result_summary (sanitized; counts/ids only)
  - FastAPI instrumentation already present via [instrument_fastapi_app()](src/observability/tracing.py:80).

Error mapping policy
- Map adapter/HTTP exceptions to canonical categories (see Error Model).
- Do not leak PII or secrets; include codes and sanitized messages only.

Call adapter binding contract
- Input validation: strict per JSON Schema before adapter execution.
- Execution: call router/service logic; ensure idempotency expectations are met.
- Output validation: strict per JSON Schema; violations → validation_error.
- Error-to-problem translation: produce standardized error payload (below).

--------------------------------------------------------------------------------

RBAC and Audit Policy

Roles per tool (registry-enforced; verified against [src/security/auth.py](src/security/auth.py))
- analysis.run: {"analyst","admin"} (router-enforced: yes)
- history.list: {"viewer","admin"} (router-enforced: yes)
- cluster.run: {"analyst","admin"} (router-enforced: no; registry enforces)
- prompts.save: {"admin"} (module-level; registry enforces)
- ingest.upload, embed.run, search.nn, reports.get, prompts.list, prompts.load: none

Audit emission (using [AuditRepository.record()](src/db/repositories/audit_repo.py:15))
- analysis.run: emitted in router; metadata includes analysis_id, prompt_version (see [analysis persistence](src/api/routers/analysis.py:282))
- cluster.run: emit with run_id, algorithm, model_name, dataset_id
- prompts.save: emit with version and metadata keys
- History/report/search/ingest/embed: no audit

Resource identifiers and metadata keys
- dataset:{id}, run:{id}, analysis:{id}, prompt:{version}
- metadata keys: { "prompt_version", "analysis_id", "run_id", "algorithm", "model_name" }

--------------------------------------------------------------------------------

Observability

Metrics
- Middleware: request_count, request_latency_seconds with labels (endpoint, method, status) via [MetricsMiddleware](src/observability/metrics.py:84)
- Domain metrics available:
  - embeddings_throughput (by model) [EMBEDDINGS_THROUGHPUT](src/observability/metrics.py:46)
  - vector_search_latency_seconds [VECTOR_SEARCH_LATENCY](src/observability/metrics.py:51)
  - analysis_token_usage [ANALYSIS_TOKEN_USAGE](src/observability/metrics.py:56)
- Registry instrumentation:
  - tool_call_count (Counter by tool_name)
  - tool_latency_seconds (Histogram by tool_name)
- Exposition: [GET /metrics](src/api/routers/metrics.py:11)

Tracing
- Initialization: [init_tracing()](src/observability/tracing.py:37)
- Instrumentation: [instrument_fastapi_app()](src/observability/tracing.py:80)
- Manual tracer accessor: [tracer()](src/observability/tracing.py:104)
- Span attributes for tool calls:
  - tool_name
  - args_summary (sanitized)
  - result_summary (sanitized)

--------------------------------------------------------------------------------

Limits and Safety

Budget and limits
- Step limit: TOOL_MAX_STEPS (e.g., 8) — abort with budget_exceeded if exceeded.
- Token budget: TOOL_TOKEN_BUDGET — cap per run; truncate sampled context or reduce k for search.
- Cost guard: configured per-run based on model token estimates; refuse if exceeded.

Configuration keys
- ANALYZER_BACKEND=openai|offline (function-calling only in openai mode)
- APP_OPENAI_API_KEY / Azure equivalents (see [OpenAI client](src/ai/llm/openai_analyzer.py:75))
- APP_ENABLE_TRACING, APP_ENABLE_METRICS, APP_ENABLE_RBAC, APP_JWT_SECRET
- Rerank backend defaults via environment as documented in README

Prompt injection defenses
- Closed-world tool whitelist: registry refuses unknown tool names.
- Strict JSON Schema validation: additionalProperties=false.
- Argument sanitization: strip/normalize strings; forbid code-like payloads.
- Refusal strategy: return validation_error with reason; do not attempt execution.

--------------------------------------------------------------------------------

Error Model

Canonical categories
- validation_error: Input/output schema violations or missing required fields.
- rbac_denied: Missing/invalid token or insufficient roles.
- budget_exceeded: Step/token/cost caps hit.
- tool_unavailable: Backend or optional dependency unavailable (e.g., HDBSCAN not installed).
- downstream_error: Router/service/adapter error (HTTP 5xx/4xx mapped).

Problem response payload (returned to analyzer loop)
{
  "error": {
    "category": "validation_error" | "rbac_denied" | "budget_exceeded" | "tool_unavailable" | "downstream_error",
    "message": "sanitized, user-safe description",
    "details": { "tool_name": "string", "endpoint": "string?", "status": 4xx|5xx?, "hint": "string?" }
  }
}

Mapping guidance
- HTTPException.status_code:
  - 400 → validation_error
  - 401/403 → rbac_denied
  - 404/409 → downstream_error (resource/index mismatch)
  - 422 → tool_unavailable (e.g., HDBSCAN not present) or validation_error depending on cause
  - 500/502/503 → downstream_error
- Budget policies → budget_exceeded

--------------------------------------------------------------------------------

Rollout and Testing Hooks

Unit tests
- Registry:
  - Name whitelist enforcement
  - Input/output JSON Schema validation
  - RBAC denial paths using [require_roles()](src/security/auth.py:67)
- Audit:
  - Emit records via [AuditRepository.record()](src/db/repositories/audit_repo.py:15) with minimal metadata

Integration tests
- Deterministic offline sequences:
  - search.nn → analysis.run → reports.get (use [OfflineAnalyzer.analyze()](src/ai/llm/offline_analyzer.py:72))
- Error paths:
  - Unknown tool name → validation_error
  - Exceeded step budget → budget_exceeded
  - RBAC denial on analysis.run/history.list → rbac_denied
  - Token budget overflow → budget_exceeded and truncation verified

CI toggles
- Offline determinism:
  - ANALYZER_BACKEND=offline; APP_EMBED_BACKEND=builtin; APP_RERANK_BACKEND=builtin
- Tracing disabled; metrics optional
- Ensure tests do not require network or provider SDKs

--------------------------------------------------------------------------------

Appendix — Alignment Confirmations

- Router existence and mapping confirmed:
  - Ingest: [upload_dataset()](src/api/routers/ingest.py:18)
  - Embed: [run_embeddings()](src/api/routers/embed.py:19)
  - Search: [knn_search()](src/api/routers/search.py:34)
  - Cluster: [run_clustering()](src/api/routers/cluster.py:46)
  - Analysis: [run_analysis()](src/api/routers/analysis.py:128)
  - Reports: [get_dataset_report()](src/api/routers/reports.py:107)
  - History: [list_analyses_history()](src/api/routers/history.py:30)
- RBAC guards present:
  - Analysis: [require_roles({"analyst","admin"})](src/api/routers/analysis.py:155)
  - History: [require_roles({"viewer","admin"})](src/api/routers/history.py:34)
- Audit emission present:
  - Analysis: [AuditRepository.record()](src/db/repositories/audit_repo.py:15) during run completion

End of Specification.
