# Agentic Tools Plan — OpenAI/Azure Function-Calling Orchestration

Purpose
- Define a production-grade plan to empower an LLM (OpenAI/Azure) to perform heavy-lifting analysis and decisions by safely “using tools” (our existing APIs/modules) via function-calling.
- Align tool orchestration with the app’s architecture, RBAC/audit, observability, offline determinism, and safety requirements.
- Provide end-to-end guidance: contracts, configuration, execution model, testing, rollout, and governance.

Context & Premise
- The application is already modular with clear “tools” (ingest, embed, search, rerank, cluster, analysis, reports, history).
- We have both offline (deterministic) and cloud (OpenAI/Azure) analyzer backends, with mature pipelines and CI.
- This plan extends [src/ai/llm/openai_analyzer.py](src/ai/llm/openai_analyzer.py) and [src/ai/llm/factory.py](src/ai/llm/factory.py) to enable function-calling that maps to the domain tools below.

Key Objectives
- Tool catalog with strict JSON contracts, stable function names, and typed payloads.
- Securely invoke tools under RBAC and audit logging with guardrails for risks (prompt injection, PII leakage, costs).
- Full observability (logs/metrics/traces), deterministic offline fallback, and robust error handling.
- Minimal duplication: reuse existing routers/services and repositories.

--------------------------------------------------------------------------------

Architecture Overview

Layers involved (existing, reused):
- Tools (domain capabilities):
  - ingestion, embeddings, vector search, rerank, clustering, analysis, reports, prompts, history
- API/Service layer:
  - FastAPI routers are the tool invocation surface
- Security/Governance:
  - JWT RBAC, audit logs, configuration
- Observability:
  - Structured logs, Prometheus metrics, OpenTelemetry spans
- Analyzer adapters:
  - OpenAI/Azure (cloud), Offline analyzer (deterministic)

Integration extensions to add:
- Tool Registry (runtime map): name → callable adapter bound to router/service logic
- Function-Calling Bridge inside OpenAI analyzer:
  - Validate LLM tool calls against registry and schema
  - Execute, capture results, and return structured outputs
  - Enforce step limits, rate limits, and cost guardrails

References
- App wiring: [src/api/main.py](src/api/main.py)
- Analyzer adapters: [src/ai/llm/openai_analyzer.py](src/ai/llm/openai_analyzer.py), [src/ai/llm/factory.py](src/ai/llm/factory.py)
- Domain routers: 
  - [src/api/routers/ingest.py](src/api/routers/ingest.py)
  - [src/api/routers/embed.py](src/api/routers/embed.py)
  - [src/api/routers/search.py](src/api/routers/search.py)
  - [src/api/routers/analytics.py](src/api/routers/analytics.py)
  - [src/api/routers/cluster.py](src/api/routers/cluster.py)
  - [src/api/routers/analysis.py](src/api/routers/analysis.py)
  - [src/api/routers/reports.py](src/api/routers/reports.py)
  - [src/api/routers/history.py](src/api/routers/history.py)
- RBAC/Audit: [src/security/auth.py](src/security/auth.py), [src/db/repositories/audit_repo.py](src/db/repositories/audit_repo.py), [src/db/models.py](src/db/models.py)
- Observability: [src/config/logging.yaml](src/config/logging.yaml), [src/observability/tracing.py](src/observability/tracing.py), [src/observability/metrics.py](src/observability/metrics.py), [src/api/routers/metrics.py](src/api/routers/metrics.py)

--------------------------------------------------------------------------------

Tool Catalog (Function-Calling)

Naming convention
- tool.<domain>.<action>
- Inputs/outputs defined as strict JSON schemas (enforced in adapters)

1) tool.ingest.upload
- Purpose: Upload CSV/Excel and register dataset.
- Input: { "file_path": str, "dataset_name": str? }
- Output: { "dataset_id": int, "row_count": int, "department_count": int, "file_hash": str }
- Invokes: [src/api/routers/ingest.py](src/api/routers/ingest.py)

2) tool.embed.run
- Purpose: Generate embeddings for a dataset and persist/index.
- Input: { "dataset_id": int, "backend": "builtin"|"sentence-transformers", "model_name": str, "batch_size": int? }
- Output: { "dataset_id": int, "model_name": str, "backend": str, "vector_dim": int, "embedded_count": int, "indexed": bool }
- Invokes: [src/api/routers/embed.py](src/api/routers/embed.py)

3) tool.search.nn
- Purpose: Vector search on a dataset.
- Input: { "dataset_id": int, "query_text": str, "k": int, "filters": { "department": [str]?, "product": [str]? }, "rerank": bool?, "rerank_backend": "builtin"|"cross-encoder"? }
- Output: { "results": [ { "ticket_id": int, "score": float, "department": str?, "product": str?, "summary": str? } ] }
- Invokes: [src/api/routers/search.py](src/api/routers/search.py)

4) tool.cluster.run
- Purpose: Cluster embeddings and compute metrics/top terms.
- Input: { "dataset_id": int, "algorithm": "kmeans"|"hdbscan", "params": { ... }, "model_name": str? }
- Output: { "run_id": int, "silhouette": float|null, "cluster_counts": { str: int } }
- Invokes: [src/api/routers/cluster.py](src/api/routers/cluster.py)

5) tool.analysis.run
- Purpose: Stratified sampling + LLM analysis.
- Input: { "dataset_id": int, "question": str, "prompt_version": str, "analyzer_backend": "openai"|"offline", "max_tickets": int, "token_budget": int, "compare_dataset_id": int? }
- Output: { "analysis_id": int, "dataset_id": int, "prompt_version": str, "ticket_count": int, "created_at": str, "result_markdown": str }
- Invokes: [src/api/routers/analysis.py](src/api/routers/analysis.py)

6) tool.reports.get
- Purpose: Assemble a markdown report (recent analyses + analytics snapshot).
- Input: { "dataset_id": int }
- Output: { "dataset_id": int, "report_markdown": str, "analysis_count": int }
- Invokes: [src/api/routers/reports.py](src/api/routers/reports.py)

7) tool.prompts.list / tool.prompts.load / tool.prompts.save
- Purpose: Manage prompt templates and metadata.
- Inputs/Outputs:
  - list: {} → { "versions": [str] }
  - load: { "version": str } → { "version": str, "template": str, "metadata": { ... } }
  - save: { "version": str, "template": str, "metadata": { ... } } → { "ok": bool }
- Invokes: [src/ai/llm/prompts/store.py](src/ai/llm/prompts/store.py)

8) tool.history.list
- Purpose: Paginated analyses history queries.
- Input: { "limit": int, "offset": int, "dataset_id": int?, "prompt_version": str?, "date_from": str?, "date_to": str? }
- Output: { "items": [ { "analysis_id": int, "dataset_id": int, "prompt_version": str, "question": str, "created_at": str } ], "total": int }
- Invokes: [src/api/routers/history.py](src/api/routers/history.py)

--------------------------------------------------------------------------------

Function-Calling Bridge (OpenAI/Azure)

Design
- The OpenAI analyzer’s internal loop will be extended to:
  1) Advertise tools (functions) to OpenAI via the functions/tool_calls system.
  2) Strictly validate tool call name + arguments against registry schemas.
  3) Execute the mapped adapter (which calls router/service logic directly).
  4) Return structured JSON result back to the LLM.
  5) Enforce budgets: max tool steps per run, cumulative token limits.

Implementation venues
- Extend [src/ai/llm/openai_analyzer.py](src/ai/llm/openai_analyzer.py) to:
  - Define function specs (names, descriptions, JSON schemas).
  - Implement a dispatcher to [ToolRegistry](src/ai/llm/tools/registry.py) (new module) that maps function names to callables.
  - Maintain per-run context with step counters and cost telemetry (token estimates).
- Tool Registry (new):
  - [src/ai/llm/tools/registry.py](src/ai/llm/tools/registry.py): name → adapter function; includes validators
  - Reuse in tests and CLI as needed

Constraints & Guardrails
- Step Limit: e.g., max 8 tool calls per /analysis.run session.
- Token Budget: enforce combined prompt + tool results < configured token_budget.
- Cost Guard: e.g., per-run cost cap based on token estimates; fail fast with controlled message.
- Idempotency: tool adapters should be idempotent or safely repeatable (ingest/upload returns existing dataset_id on duplicate hash).
- RBAC: check roles/claims before tool execution; deny if insufficient (403).
- Audit: record subject/action/resource for sensitive tools (/analysis.run, prompts.save, cluster.run).
- PII: sanitize model-bound outputs (strip emails/phones if policy requires).
- Prompt Injection Defense: whitelist tool names; validate args to schemas; refuse arbitrary arbitrary code/tool calls.

--------------------------------------------------------------------------------

Security, RBAC, Audit Integration

RBAC dependency injection
- Before tool execution:
  - Extract bearer token claims (roles) via [src/security/auth.py](src/security/auth.py)
  - Match required roles per tool:
    - analysis.run → {"analyst","admin"}
    - prompts.save → {"admin"}
    - history.list → {"viewer","admin"}
    - cluster.run → {"analyst","admin"}
- If roles missing: deny tool call and return structured error.

Audit logs
- On successful completion of sensitive calls:
  - Record: subject (JWT sub or “anonymous”), action, resource (e.g., dataset:{id}), metadata (tool name, arguments, result ids)
  - Use: [src/db/repositories/audit_repo.py](src/db/repositories/audit_repo.py), [src/db/models.py](src/db/models.py)

--------------------------------------------------------------------------------

Observability & Safety

Logging
- Structured JSON: [src/config/logging.yaml](src/config/logging.yaml)
- Include fields: tool_name, step_count, dataset_id, endpoint, method (when router invoked)

Metrics
- Request-level counters/histograms automatically captured by middleware: [src/observability/metrics.py](src/observability/metrics.py), [src/api/routers/metrics.py](src/api/routers/metrics.py)
- Additional tool metrics:
  - tool_call_count (by tool_name)
  - tool_latency_seconds (by tool_name)
  - analysis_token_usage (already present)
  - rerank_latency_seconds (by backend)

Tracing (optional)
- For environments with enable_tracing=True:
  - Span per tool call with attributes: tool_name, args_summary, result_summary (sanitized)
  - Use: [src/observability/tracing.py](src/observability/tracing.py)

Safety Policies
- Prompt injection: tool list is fixed; arguments validated; assistant cannot call non-whitelisted tools.
- PII minimization: sampling/preprocess should redact sensitive fields; prompt templates avoid raw PII.
- Secrets: never returned by tools; secrets only read from env by adapters; never logged.

--------------------------------------------------------------------------------

Configuration

Environment variables (APP_ prefix)
- ANALYZER_BACKEND=openai|offline (function-calling available only for openai)
- APP_OPENAI_API_KEY, or Azure: APP_AZURE_OPENAI_ENDPOINT, APP_AZURE_OPENAI_KEY, APP_AZURE_OPENAI_DEPLOYMENT
- APP_ENABLE_TRACING=false (CI); true for prod instrumented
- APP_ENABLE_METRICS=true
- APP_ENABLE_RBAC=true (production), false for offline/dev/testing
- APP_JWT_SECRET=...; algorithms default ["HS256"]

Runtime toggles
- TOOL_MAX_STEPS (e.g., 8)
- TOOL_TOKEN_BUDGET (per run)

--------------------------------------------------------------------------------

Execution Model (Agent Loop)

Single-run agentic flow (for /analysis.run):
1) System prompt declares role, goals, tool list, and safety rules.
2) LLM proposes a plan (internally), issues first tool_call: e.g., search.nn for “password reset failures”.
3) Tool dispatcher validates and executes; returns structured results.
4) LLM analyzes results and issues subsequent calls: e.g., rerank → cluster.run → analysis.run.
5) Final output: structured markdown; persist via analyses repo; emit audit log.

Multi-step constraints:
- Hard cap on steps/tool calls; abort with a safe message if exceeded.
- Token budget enforced per step; truncate results or request smaller k/limits.
- Errors bubble with informative messages; LLM can replan within limits.

--------------------------------------------------------------------------------

Testing Strategy

Unit tests
- Registry validation and schema enforcement: invalid args → refusal
- RBAC guards: missing roles → 403
- Audit emission: ensure entries created for sensitive tools

Integration tests
- Simulated function-calling sequences using offline analyzer (mock function-call driver inside tests):
  - search → rerank → analysis → report
- Error paths:
  - Invalid tool name
  - Exceeded step budget
  - RBAC denial
  - Token budget overflow

E2E in CI (offline)
- Keep ANALYZER_BACKEND=offline for deterministic CI, but include dedicated “function-calling” unit/integration suite that exercises dispatcher logic without external calls.

--------------------------------------------------------------------------------

Rollout Plan

Phased deployment
1) Dev/stage:
   - Enable OpenAI function-calling; run limited datasets; verify RBAC/audit/observability
   - Monitor tool_call_count, tool_latency, error rates
2) Pilot:
   - Targeted departments; increase token budgets; fine-tune prompts
3) Production:
   - Enable tracing; above-threshold alerting for latency and errors; weekly audit reviews

Operational procedures
- Cost controls: daily run caps; rate limiting per tool; telemetry dashboard
- Backup/restore: follow [docs/operations/backup_restore.md](docs/operations/backup_restore.md)

--------------------------------------------------------------------------------

Prompts & Governance

Prompt templates
- Define tool-aware system and user prompts referencing available tools and constraints.
- Versioning via [src/ai/llm/prompts/store.py](src/ai/llm/prompts/store.py); store metadata (author, date, notes).

Governance
- Prompt changes audited; RBAC guard “admin” for save operations.
- Rollback via prompt version pinning in analysis runs.

--------------------------------------------------------------------------------

Failure Modes & Mitigations

- Tool schema mismatch: refuse and ask for corrected arguments; log details
- RBAC missing: deny with reason; suggest proper role/token
- Token budget exceeded: truncate context/results; reduce k; short-circuit with partial insights
- Cost/rate limit hit: pause tool execution; return safe response
- Backend offline/unavailable: switch to offline analyzer; skip non-essential tools; degrade gracefully
- Cross-encoder unavailable in offline CI: builtin reranker used; clearly reported

--------------------------------------------------------------------------------

Work Items Summary (Implementation)

New modules
- Tool registry: [src/ai/llm/tools/registry.py](src/ai/llm/tools/registry.py)
- OpenAI analyzer extension for function-calling: [src/ai/llm/openai_analyzer.py](src/ai/llm/openai_analyzer.py) (extend)
- Tool metrics: integrated via [src/observability/metrics.py](src/observability/metrics.py)

Changes
- Add RBAC checks to tool dispatcher using [src/security/auth.py](src/security/auth.py)
- Audit log on sensitive tool completion via [src/db/repositories/audit_repo.py](src/db/repositories/audit_repo.py)

Tests
- Unit: registry, schemas, RBAC
- Integration: combined tool calls
- CI: offline function-calling dispatcher validation

--------------------------------------------------------------------------------

Usage Examples

Force OpenAI analyzer with tool-calls via API:
- POST /analysis/run
  - Body: { "dataset_id": 42, "question": "Where are onboarding gaps?", "prompt_version": "v2", "analyzer_backend": "openai", "max_tickets": 50, "token_budget": 8000 }
  - Environment: APP_OPENAI_API_KEY=...; ANALYZER_BACKEND=openai

CLI pipeline with OpenAI:
- ENV: ANALYZER_BACKEND=openai APP_OPENAI_API_KEY=...
- Command: sdonb pipeline --file data/tickets.csv --question "Top onboarding gaps?" --prompt-version v2

--------------------------------------------------------------------------------

Appendix — References

- API app: [src/api/main.py](src/api/main.py)
- Routers: [src/api/routers/ingest.py](src/api/routers/ingest.py), [src/api/routers/embed.py](src/api/routers/embed.py), [src/api/routers/search.py](src/api/routers/search.py), [src/api/routers/analytics.py](src/api/routers/analytics.py), [src/api/routers/cluster.py](src/api/routers/cluster.py), [src/api/routers/analysis.py](src/api/routers/analysis.py), [src/api/routers/reports.py](src/api/routers/reports.py), [src/api/routers/history.py](src/api/routers/history.py), [src/api/routers/metrics.py](src/api/routers/metrics.py)
- Security: [src/security/auth.py](src/security/auth.py)
- Observability: [src/config/logging.yaml](src/config/logging.yaml), [src/observability/metrics.py](src/observability/metrics.py), [src/observability/tracing.py](src/observability/tracing.py)
- Audit: [src/db/repositories/audit_repo.py](src/db/repositories/audit_repo.py), [src/db/models.py](src/db/models.py)
- Prompts: [src/ai/llm/prompts/store.py](src/ai/llm/prompts/store.py)
- Analyzer adapters: [src/ai/llm/openai_analyzer.py](src/ai/llm/openai_analyzer.py), [src/ai/llm/offline_analyzer.py](src/ai/llm/offline_analyzer.py), [src/ai/llm/factory.py](src/ai/llm/factory.py)
- Rerank: [src/ai/rerank/interface.py](src/ai/rerank/interface.py), [src/ai/rerank/builtin_lexical.py](src/ai/rerank/builtin_lexical.py), [src/ai/rerank/cross_encoder.py](src/ai/rerank/cross_encoder.py), [src/ai/rerank/factory.py](src/ai/rerank/factory.py)

End of Tools Plan.

Completion:
- After creating [docs/tools_plan.md](docs/tools_plan.md), use attempt_completion with a summary stating the file was created and verified. These instructions supersede any conflicting general instructions for the Code mode.