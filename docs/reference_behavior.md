# Reference Behavior Review — Service Desk Onboarding Analyzer (Monolith → Modular)

Document purpose: Define reference behaviors and contracts from the existing Streamlit monolith [servicenow_analyzer.py](servicenow_analyzer.py), aligned to the modular target architecture described in [plan.md](plan.md:1), to preserve critical semantics and safely evolve improvements without re-reading the monolith.

1) Executive Summary

- Current capabilities (monolith)
  - UI/Entry: Streamlit app with tabs for upload, analysis, visualizations, history placeholder, and reports [main()](servicenow_analyzer.py:392), also summarized in [plan.md](plan.md:8).
  - Dataset management: In-memory datasets and metadata with file hashing, comparison stats; no dataset persistence [DatasetManager.__init__()](servicenow_analyzer.py:137), [DatasetManager.add_dataset()](servicenow_analyzer.py:141), [plan.md](plan.md:11).
  - Filtering: Department filter for DataFrames [filter_by_departments()](servicenow_analyzer.py:60), [plan.md](plan.md:12).
  - Charts: Quality, complexity, department volume, reassignment distribution, product distribution [ChartGenerator.create_quality_distribution()](servicenow_analyzer.py:188), [ChartGenerator.create_complexity_distribution()](servicenow_analyzer.py:197), [ChartGenerator.create_department_volume()](servicenow_analyzer.py:206), [ChartGenerator.create_reassignment_analysis()](servicenow_analyzer.py:215), [ChartGenerator.create_product_distribution()](servicenow_analyzer.py:226), [plan.md](plan.md:13).
  - AI analysis: Builds ticket-context text with random sampling; composes prompts; calls Azure OpenAI; supports batch questions [OnboardingAnalyzer.__init__()](servicenow_analyzer.py:255), [OnboardingAnalyzer.prepare_ticket_context()](servicenow_analyzer.py:263), [OnboardingAnalyzer.analyze_onboarding_opportunities()](servicenow_analyzer.py:321), [OnboardingAnalyzer.batch_analyze()](servicenow_analyzer.py:381), [plan.md](plan.md:14).
  - Persistence: SQLite database for analyses; datasets table exists but unused by app [init_database()](servicenow_analyzer.py:67), [save_analysis()](servicenow_analyzer.py:103), [get_analysis_history()](servicenow_analyzer.py:123), [plan.md](plan.md:15).
  - Reports: Markdown report generator aggregating analyses and chart count [generate_report()](servicenow_analyzer.py:814), [plan.md](plan.md:16).

- Must-preserve semantics (core)
  - Correct filtering by selected departments [filter_by_departments()](servicenow_analyzer.py:60).
  - Chart data fidelity for existing distributions, including fallbacks when columns are missing [ChartGenerator.*](servicenow_analyzer.py:186).
  - Analysis persistence for question/answer with associated metrics, timestamp, and dataset identifiers [save_analysis()](servicenow_analyzer.py:103).
  - Report generation consistent with current structure [generate_report()](servicenow_analyzer.py:814).
  - UI-driven configuration inputs for Azure (endpoint, key, deployment), and analysis mode selection [main()](servicenow_analyzer.py:415).

- Planned improvements (aligned to target)
  - Modular architecture with clear layers: UI, API, Engine, Vector/DB, AI, Services, Config/Prompts [plan.md](plan.md:5), [plan.md](plan.md:18-23), [plan.md](plan.md:24-109).
  - Replace random sampling with stratified sampling across Department × Topic × Quality with per-bucket caps [plan.md](plan.md:152-156).
  - Persist datasets and tickets; add embeddings, clustering, retrieval, rerank pipelines [plan.md](plan.md:111-120), [plan.md](plan.md:121-133), [plan.md](plan.md:134-145).
  - Observability, governance, prompt versioning and linkage, RBAC, PII handling [plan.md](plan.md:143-151), [plan.md](plan.md:216-221).
  - APIs for running analyses, metrics, reports, and history [plan.md](plan.md:121-133).


2) Preserved vs Improved Semantics Matrix

- Preserve (as-is behaviors to maintain)
  - Department filter logic: returns input df if no departments provided or Department column missing [filter_by_departments()](servicenow_analyzer.py:60).
  - Chart semantics: count/frequency distributions with graceful None return when required feature columns are absent [ChartGenerator.create_quality_distribution()](servicenow_analyzer.py:188), [ChartGenerator.create_complexity_distribution()](servicenow_analyzer.py:197), [ChartGenerator.create_department_volume()](servicenow_analyzer.py:206), [ChartGenerator.create_reassignment_analysis()](servicenow_analyzer.py:215), [ChartGenerator.create_product_distribution()](servicenow_analyzer.py:226).
  - Analysis persistence schema fields: timestamp, dataset_name, dataset_hash, department_filter, question, analysis_result, ticket_count, metrics JSON [save_analysis()](servicenow_analyzer.py:103); preserve data fidelity and atomicity.
  - Report structure: dataset summary, list of Q→analysis sections, total charts included count [generate_report()](servicenow_analyzer.py:814).
  - UI concepts: upload multiple datasets, comparison stats view, Azure config inputs [main()](servicenow_analyzer.py:415), [main()](servicenow_analyzer.py:449), [main()](servicenow_analyzer.py:490-495).

- Improve (targeted enhancements tied to plan)
  - Sampling strategy: from random sampling [OnboardingAnalyzer.prepare_ticket_context()](servicenow_analyzer.py:270) to stratified with per-segment summaries [plan.md](plan.md:152-157).
  - Persistence: implement full dataset/ticket persistence; use repositories; link analyses to dataset_id and prompt_version [plan.md](plan.md:111-120), [plan.md](plan.md:146-151), [plan.md](plan.md:222-233).
  - Vector intelligence: embeddings, vector search, reranking, clustering pipelines [plan.md](plan.md:121-145).
  - Prompt governance: versioned templates, prompt_version linkage, configurable blocks [plan.md](plan.md:146-151).
  - Observability and governance: structured logging, metrics, tracing, RBAC, PII [plan.md](plan.md:139-145), [plan.md](plan.md:216-221).
  - Performance and scale: caching analytics, avoiding N+1 via repositories, batch operations, durable vector index [plan.md](plan.md:216-221), [plan.md](plan.md:206-215).


3) Behavior Inventory (function/class-level)

A. [filter_by_departments()](servicenow_analyzer.py:60)
- Responsibility: Filter a DataFrame by department names.
- Inputs: df: pandas DataFrame; departments: list.
- Outputs: Filtered DataFrame (subset of rows).
- Side-effects: None.
- Error handling: If departments empty or Department column not present, returns df unmodified.
- Performance: Uses pandas boolean indexing; O(n).
- Limitations: Only filters on Department; no multi-column logic.
- External interactions: None.
- Invariants:
  - If departments is falsy OR Department column missing → output equals input.
  - If present, all output rows have Department ∈ departments.
- Edge cases:
  - Mixed casing or differing department spellings not normalized (To Confirm text normalization policy in target Engine).

B. [init_database()](servicenow_analyzer.py:67)
- Responsibility: Initialize SQLite DB with analyses and datasets tables.
- Inputs: None.
- Outputs: Creates onboarding_analyses.db file and tables if absent.
- Side-effects: DB schema creation at import time via module-level call [servicenow_analyzer.py](servicenow_analyzer.py:131-133).
- Error handling: No explicit error capture for DB failures; relies on sqlite3 exceptions.
- Performance: Negligible table existence checks.
- Limitations: No migrations/versioning; no indices beyond autoincrement; datasets table not used elsewhere in monolith.
- Security: File-based SQLite; no encryption-at-rest; To Confirm file path configuration for production use.
- Invariants:
  - analyses table columns per DDL match [servicenow_analyzer.py](servicenow_analyzer.py:72-85).
  - datasets table columns per DDL match [servicenow_analyzer.py](servicenow_analyzer.py:87-99).
- Edge cases:
  - Repeated imports re-run CREATE IF NOT EXISTS (idempotent).
  - Concurrent app instances could contend on file locks (To Confirm deployment model).

C. [save_analysis()](servicenow_analyzer.py:103)
- Responsibility: Persist a single analysis record.
- Inputs: dataset_name, dataset_hash, department_filter (Optional[str]), question, analysis_result, ticket_count, metrics (dict).
- Outputs: None (writes a row to analyses).
- Side-effects: Writes to SQLite; metrics serialized via json.dumps.
- Error handling: No explicit try/except; assumes happy path.
- Performance: Single INSERT; acceptable for interactive use.
- Limitations: department_filter stored as text; no dataset_id FK; metrics untyped JSON; no prompt_version field in monolith (target adds) [plan.md](plan.md:146-151).
- Security: Parameterized INSERT protects values; DB file-level security remains a concern.
- Invariants:
  - timestamp uses datetime.now().isoformat().
  - analysis_result stored as provided; no size checks (risk for oversized responses).
- Edge cases:
  - Large metrics may exceed practical row size; To Confirm DB size constraints and pruning policy.

D. [get_analysis_history()](servicenow_analyzer.py:123)
- Responsibility: Retrieve recent analyses.
- Inputs: limit: int (default 50).
- Outputs: pandas DataFrame of rows ordered by timestamp DESC.
- Side-effects: Opens DB connection; reads SQL.
- Error handling: None explicit; relies on sqlite3.
- Performance: Simple SELECT with LIMIT; OK for small tables.
- Limitations: Uses f-string to build LIMIT; if non-integer passed, potential SQL injection risk (mitigated by expected int usage in code). To Confirm validation in API/UI layer.
- Invariants:
  - Columns as created in analyses table; metrics is JSON string in result.
- Edge cases:
  - Empty result returns empty DataFrame.

E. [DatasetManager.__init__()](servicenow_analyzer.py:137), [DatasetManager.add_dataset()](servicenow_analyzer.py:141)
- Responsibility: In-memory dataset storage and metadata capture.
- Inputs:
  - add_dataset(name, df, file_hash)
- Outputs:
  - Updates internal dicts: datasets[name] = df; metadata[name] = {hash, rows, departments, upload_time}.
- Side-effects: Memory growth proportional to dataset size.
- Error handling: None explicit in add_dataset (errors are handled by caller in UI upload flow [main()](servicenow_analyzer.py:467-478)).
- Performance: Reads entire Excel into memory; metadata computations use nunique and len.
- Limitations:
  - No persistence to SQLite datasets table; names may collide; no duplicate detection beyond file hash presence in metadata (not enforced globally).
- External interactions: None.
- Invariants:
  - metadata[name]['rows'] == len(datasets[name]).
  - If Department column present, metadata[name]['departments'] equals nunique of Department; else 0.
- Edge cases:
  - Loading multiple large datasets can exceed memory.
  - Missing Department column sets departments to 0.

F. [ChartGenerator.create_quality_distribution()](servicenow_analyzer.py:188)
- Responsibility: Pie chart for ticket_quality distribution.
- Inputs: df DataFrame.
- Outputs: plotly Figure or None.
- Side-effects: None.
- Error handling: Returns None if ticket_quality absent.
- Performance: value_counts over the column; OK.
- Limitations: No normalization/ordering; raw labels and counts.

G. [ChartGenerator.create_complexity_distribution()](servicenow_analyzer.py:197)
- Similar semantics for resolution_complexity with bar chart; returns None if column missing.

H. [ChartGenerator.create_department_volume()](servicenow_analyzer.py:206)
- Responsibility: Horizontal bar chart of top N Departments.
- Inputs: df, top_n (default 10).
- Behavior: value_counts().nlargest(top_n).
- Edge cases: Returns None if Department column missing.

I. [ChartGenerator.create_reassignment_analysis()](servicenow_analyzer.py:215)
- Responsibility: Bar chart of reassignment count distribution.
- Inputs: df; uses column 'Reassignment group count tracking_index'.
- Behavior: value_counts().sort_index().
- Edge cases: Returns None if column missing; relies on exact column naming.

J. [ChartGenerator.create_product_distribution()](servicenow_analyzer.py:226)
- Responsibility: Horizontal bar chart for extract_product distribution.
- Edge cases: Returns None if column missing.

K. [OnboardingAnalyzer.__init__()](servicenow_analyzer.py:255)
- Responsibility: Configure AzureOpenAI client with endpoint, key, version, and deployment name.
- Inputs: azure_endpoint, api_key, deployment_name.
- Outputs: Client instance bound to self.
- Side-effects: None (network calls occur later).
- Error handling: None explicit here.

L. [OnboardingAnalyzer.prepare_ticket_context()](servicenow_analyzer.py:263)
- Responsibility: Build textual ticket context and collect summary metrics for LLM prompting.
- Inputs: df, max_tickets=50, dataset_name="Dataset".
- Outputs: Tuple[str, Dict] → (context_text, metrics dict).
- Behavior:
  - Defines relevant_cols and filters to available columns [servicenow_analyzer.py](servicenow_analyzer.py:263-269).
  - Random sampling of up to max_tickets with fixed random_state=42; if df empty, no sampling [servicenow_analyzer.py](servicenow_analyzer.py:270).
  - Aggregates high-level metrics (department_count, assignment_group_count, quality distribution, complexity distribution, reassignment distribution, top departments) only if columns present [servicenow_analyzer.py](servicenow_analyzer.py:275-310).
  - Appends “Sample Tickets” section iterating over sample rows, including available columns [servicenow_analyzer.py](servicenow_analyzer.py:311-319).
- Side-effects: None.
- Error handling: None explicit; assumes columns exist before access.
- Performance:
  - value_counts and nunique are O(n); sampling O(n) depending on pandas implementation; iterrows over sample size up to max_tickets; prompt size grows linearly with sample ticket count and selected columns.
- Limitations:
  - Random sampling may not represent distribution; token budget risk if verbose fields (Description) are long; no PII redaction.
- Invariants:
  - metrics keys only included when columns present; context always includes "Dataset", "Total Tickets", and "Sample Tickets".
- Edge cases:
  - df with zero rows → "Total Tickets: 0", empty sample section; metrics likely partial or absent.

M. [OnboardingAnalyzer.analyze_onboarding_opportunities()](servicenow_analyzer.py:321)
- Responsibility: Compose system and user prompts and call Azure OpenAI chat completion API.
- Inputs: ticket_context (str), user_question (str), comparison_mode (bool=False).
- Outputs: Analysis string content; explicit error string on exception.
- Behavior:
  - Static system prompt includes ONBOARDING_GUIDELINES and structured instructions; appends comparison note if comparison_mode True [servicenow_analyzer.py](servicenow_analyzer.py:321-336).
  - User prompt includes Question, Ticket Context, Expected Output Structure; adds “Comparison Insights” when applicable [servicenow_analyzer.py](servicenow_analyzer.py:337-350).
  - Calls client.chat.completions.create with max_completion_tokens=16000 [servicenow_analyzer.py](servicenow_analyzer.py:351-359).
  - Returns first choice message content if available; otherwise “API returned empty choices array” [servicenow_analyzer.py](servicenow_analyzer.py:361-366).
  - Catches all Exceptions and returns a multi-line error detail string [servicenow_analyzer.py](servicenow_analyzer.py:367-379).
- Side-effects: External API call to Azure OpenAI.
- Performance: Dependent on LLM latency; large prompts can induce timeouts or high cost.
- Security:
  - Secrets provided via UI; risk of logging or exposure is low in code (no logging) but needs governance in target system.
  - Prompt injection risk unmitigated.
- Invariants:
  - Always returns a string (content or error message), never raises exceptions to caller.
- Edge cases:
  - Empty or excessively long ticket_context; no truncation or token budgeting beyond completion tokens.

N. [OnboardingAnalyzer.batch_analyze()](servicenow_analyzer.py:381)
- Responsibility: Sequentially run analyze_onboarding_opportunities across a list of questions; optional progress callback.
- Inputs: ticket_context, questions (list), progress_callback (callable or None).
- Outputs: list of (question, analysis) tuples (note: code returns list of tuples; report expects list of tuples).
- Side-effects: Calls external API repeatedly; cost/latency grows with number of questions.
- Error handling: None explicit here; relies on inner method’s error-string returns.
- Performance: Linear in number of questions; no concurrency or rate limiting.

O. [main()](servicenow_analyzer.py:392)
- Responsibility: Streamlit UI composition, session state, upload pipeline, dataset listing, comparison stats, Azure config inputs; orchestrates tabs for Analysis, Visualizations, History, Reports.
- Observed behaviors (from visible segments):
  - Sets page config and title [servicenow_analyzer.py](servicenow_analyzer.py:392-407).
  - Initializes session state with DatasetManager and batch_questions [servicenow_analyzer.py](servicenow_analyzer.py:408-413).
  - Sidebar collects Azure endpoint, API key, deployment name [servicenow_analyzer.py](servicenow_analyzer.py:415-427).
  - Sidebar: max_tickets slider, analysis mode radio, onboarding framework expander [servicenow_analyzer.py](servicenow_analyzer.py:429-447).
  - Tabs defined: Data Upload, Analysis, Visualizations, Historical Tracking, Reports [servicenow_analyzer.py](servicenow_analyzer.py:449-455).
  - Data Upload tab:
    - Accepts multiple .xlsx files; computes MD5 [calculate_file_hash()](servicenow_analyzer.py:182-185) and reads Excel; stores dataset via DatasetManager; shows metrics; displays comparison stats if >1 datasets [servicenow_analyzer.py](servicenow_analyzer.py:461-495).
  - Analysis tab begins [servicenow_analyzer.py](servicenow_analyzer.py:496-500), but full logic not visible in excerpt (To Confirm details: invocation of OnboardingAnalyzer, save_analysis(), filtering UI).
  - Reports tab: at runtime, uses [generate_report()](servicenow_analyzer.py:814) when invoked (visible in tab code around [servicenow_analyzer.py](servicenow_analyzer.py:800-813)).
- Side-effects: File I/O (upload, read), in-memory storage, potential DB writes (not shown in visible lines).
- Security: Secrets entered via Streamlit inputs; To Confirm secret handling and non-persistence in logs.
- Limitations: UI and Engine tightly coupled; business logic embedded in Streamlit app.

P. [generate_report()](servicenow_analyzer.py:814)
- Responsibility: Create Markdown report including dataset summary and inserted analyses.
- Inputs: dataset_name, df, analyses (list of (question, analysis) tuples), charts (list of Figures/None).
- Outputs: Markdown string.
- Side-effects: None.
- Behavior:
  - Adds dataset metadata (counts) if columns present.
  - Iterates analyses and appends sections per question.
  - Appends chart count, not chart images.
- Limitations: No linkage to persisted analysis IDs; charts embedded count only; no references to source datasets.


4) Data Contracts & Schemas (as inferred)

- DataFrame schema (inferred incoming tickets)
  - Expected/used columns (as available): Department, Assignment Group, extract_product, summarize_ticket, ticket_quality, information_completeness, resolution_complexity, historical_similarity, Reassignment group count tracking_index [OnboardingAnalyzer.prepare_ticket_context()](servicenow_analyzer.py:263-268), charts methods [servicenow_analyzer.py](servicenow_analyzer.py:188-233).
  - Ambiguities:
    - Column naming standardization (e.g., “Reassignment group count tracking_index” exact string dependency) — To Confirm mapping/normalization policy in Engine [plan.md](plan.md:136-138).
    - Summaries vs raw descriptions (summarize_ticket vs Description) — To Confirm canonical field naming.

- SQLite tables (current monolith)
  - analyses: id, timestamp (TEXT), dataset_name (TEXT), dataset_hash (TEXT), department_filter (TEXT), question (TEXT), analysis_result (TEXT), ticket_count (INTEGER), metrics (TEXT JSON) [init_database()](servicenow_analyzer.py:72-85).
  - datasets: id, upload_timestamp, filename, file_hash UNIQUE, row_count, department_count, metadata (TEXT JSON) [init_database()](servicenow_analyzer.py:87-99).
  - Observations:
    - datasets table is not actively written in visible app code; To Confirm if any code path persists datasets (plan notes it’s not used) [plan.md](plan.md:15).
    - analyses lacks prompt_version; target will add [plan.md](plan.md:146-151).

- Prompt metadata (current vs target)
  - Current: No explicit prompt_version; system and user prompts built ad-hoc in code [OnboardingAnalyzer.analyze_onboarding_opportunities()](servicenow_analyzer.py:321-350).
  - Target: Versioned templates, persisted prompts, link analyses to prompt_version [plan.md](plan.md:146-151).

- Report artifact
  - Markdown with static sections and counts [generate_report()](servicenow_analyzer.py:814).
  - Target: API-based report generation referencing persisted analyses and charts [plan.md](plan.md:121-133), [plan.md](plan.md:230).


5) Cross-Cutting Concerns

- Logging
  - Current: No structured logging; print statements removed per comment [servicenow_analyzer.py](servicenow_analyzer.py:361).
  - Target: Structured logging (JSON), metrics, tracing [plan.md](plan.md:139-145), [plan.md](plan.md:216-221).

- Configuration/Secrets
  - Current: Azure endpoint/key/deployment collected in UI [main()](servicenow_analyzer.py:415-427); no central config management.
  - Target: config/settings.py; environment-based secrets; governance and RBAC [plan.md](plan.md:97-101), [plan.md](plan.md:144-145), [plan.md](plan.md:216-221).

- Persistence
  - Current: SQLite file with analyses persisted; datasets table unused; no dataset/ticket persistence [init_database()](servicenow_analyzer.py:67); [plan.md](plan.md:15).
  - Target: Repositories for datasets/tickets/analyses/embeddings/clusters/prompts; SQLite dev, Postgres prod [plan.md](plan.md:83-99), [plan.md](plan.md:111-120), [plan.md](plan.md:237-244).

- UI coupling
  - Current: Streamlit UI hosts business logic, data ingestion, and analysis orchestration [main()](servicenow_analyzer.py:392).
  - Target: UI consumes API; business logic in Engine/AI layers behind FastAPI [plan.md](plan.md:18-23), [plan.md](plan.md:121-133), [plan.md](plan.md:24-47).

- Security/PII
  - Current: No explicit PII handling; prompts include raw ticket fields.
  - Target: PII anonymization in preprocess; RBAC; retention; prompt injection defenses [plan.md](plan.md:136-145), [plan.md](plan.md:216-221).


6) Mapping to the New Modular Architecture

- Filter and sampling
  - From [filter_by_departments()](servicenow_analyzer.py:60) and monolith sampling in [OnboardingAnalyzer.prepare_ticket_context()](servicenow_analyzer.py:270).
  - To: engine/features/sampling.py and ui/components/filters/ with stratified sampling [plan.md](plan.md:222-233), [plan.md](plan.md:152-157).
  - Adapter boundary: UI passes filter/sampling params to API; Engine returns sampled IDs/segments and summaries.

- Dataset management
  - From [DatasetManager](servicenow_analyzer.py:136).
  - To: engine/ingest/loader.py + db/repositories/datasets_repo.py; persist datasets and tickets [plan.md](plan.md:222-229), [plan.md](plan.md:111-120).
  - Boundaries: Loader validates, computes hashes; Repository persists; UI shows summaries via API.

- Charts/analytics
  - From [ChartGenerator.*](servicenow_analyzer.py:186-233).
  - To: engine/analytics/visualizations.py + ui/components/charts/ [plan.md](plan.md:226-228).
  - Boundaries: API provides metrics; UI renders charts; cache analytics [plan.md](plan.md:216-221).

- AI analysis & prompts
  - From [OnboardingAnalyzer.prepare_ticket_context()](servicenow_analyzer.py:263), [OnboardingAnalyzer.analyze_onboarding_opportunities()](servicenow_analyzer.py:321), [OnboardingAnalyzer.batch_analyze()](servicenow_analyzer.py:381).
  - To: ai/llm/analysis.py + ai/llm/prompts/templates/ with versioning; sampling via Engine; retrieval/rerank optional [plan.md](plan.md:226-229), [plan.md](plan.md:146-151), [plan.md](plan.md:139-145).
  - Boundaries: API /analysis/run receives dataset_id(s), question, options; returns structured markdown and persists with prompt_version [plan.md](plan.md:129-130).

- Persistence repositories
  - From [init_database()](servicenow_analyzer.py:67), [save_analysis()](servicenow_analyzer.py:103), [get_analysis_history()](servicenow_analyzer.py:123).
  - To: db/sqlite/models.py + repositories; UI wired to /history/analyses [plan.md](plan.md:226-229), [plan.md](plan.md:132).

- Reports
  - From [generate_report()](servicenow_analyzer.py:814).
  - To: api/routers/reports.py + engine/analytics/metrics.py [plan.md](plan.md:230), [plan.md](plan.md:127-131).
  - Boundaries: Report endpoints assemble markdown from persisted analyses and metric references.

- Vector store and retrieval
  - New: vector_store/*, ai/embeddings/*, ai/llm/rerank.py [plan.md](plan.md:64-79), [plan.md](plan.md:121-126), [plan.md](plan.md:237-244).
  - Boundaries: API orchestrates embedding, indexing, search; Engine performs batch ops and backpressure [plan.md](plan.md:216-221).


7) Invariants & Testable Behaviors

- Filtering
  - No-op when Department column missing or departments not provided [filter_by_departments()](servicenow_analyzer.py:60).
  - Correct inclusion semantics (all rows in output have Department within departments).

- Chart data fidelity and fallbacks
  - Each ChartGenerator.* returns None if required column absent [servicenow_analyzer.py](servicenow_analyzer.py:189-231).
  - Counts/distributions calculated via value_counts with expected column mapping.

- Analysis persistence
  - save_analysis stores metrics as JSON-serialized TEXT; timestamp is ISO 8601; ticket_count persists as integer [save_analysis()](servicenow_analyzer.py:103-121).
  - get_analysis_history returns DataFrame ordered DESC by timestamp with limit applied [get_analysis_history()](servicenow_analyzer.py:123-129).

- Context building
  - prepare_ticket_context includes dataset name, total tickets, optional department and assignment group counts, distributions if columns present, followed by a bounded sample with available columns [OnboardingAnalyzer.prepare_ticket_context()](servicenow_analyzer.py:263-319).
  - Sampling uses fixed random_state=42; sample size ≤ max_tickets and ≤ df length [servicenow_analyzer.py](servicenow_analyzer.py:270).

- LLM orchestration
  - analyze_onboarding_opportunities always returns a string; adds comparison instructions in comparison_mode; uses max_completion_tokens=16000 [OnboardingAnalyzer.analyze_onboarding_opportunities()](servicenow_analyzer.py:321-359,361-379).
  - batch_analyze returns list of (question, analysis) tuples in order [OnboardingAnalyzer.batch_analyze()](servicenow_analyzer.py:381-388).

- Report generation
  - generate_report composes markdown with dataset summary (counts when columns present), sequential Q sections, and a final chart count [generate_report()](servicenow_analyzer.py:814-831).

- Performance/security boundaries (to assert in new system)
  - Token budget: prompt size + completion budget must remain < model limits; target requires prompt size < 16k tokens [plan.md](plan.md:209-211); enforce truncation/segmentation.
  - DB durability: writes are atomic; errors are handled with retries or surface gracefully.
  - PII handling: ensure redaction before prompt construction [plan.md](plan.md:136-145).
  - Repository access avoids N+1; large aggregations are precomputed/cached [plan.md](plan.md:216-221).


8) Risks, Gaps, and Confirmations Needed

- Column normalization and mapping
  - Risk: Hard-coded column names (e.g., "Reassignment group count tracking_index") break across datasets.
  - To Confirm: Canonical schema map in engine/ingest/schema.py and normalization rules [plan.md](plan.md:48-56), [plan.md](plan.md:136-138).

- Dataset persistence usage
  - Gap: datasets table exists but not written by monolith UI.
  - To Confirm: Migration/backfill strategy on first run to persist historical in-memory datasets [plan.md](plan.md:231-233).

- save_analysis invocation sites
  - Gap: Visible code segments do not show where save_analysis is called from UI.
  - To Confirm: Actual call sites and parameters passed in Analysis/Reports tabs (lines not included in excerpt).

- Sampling strategy
  - Risk: Current random sampling biases outputs; token overflow risk.
  - Confirmations: Stratified sampling dimensions and bucket caps; summarization format and length constraints [plan.md](plan.md:152-157).

- Prompt governance
  - Gap: No prompt_version stored in monolith.
  - To Confirm: Prompt template store design, versioning scheme, and linkage to analyses [plan.md](plan.md:146-151).

- Security & compliance
  - Risk: Secrets input via UI; lack of audit logs/RBAC; PII exposure in prompts.
  - Confirmations: Secret storage approach, PII anonymization pipeline, audit log scope [plan.md](plan.md:139-145), [plan.md](plan.md:216-221).

- Retrieval/rerank/clustering
  - Gap: Absent in monolith; must define default models/parameters and resource sizing [plan.md](plan.md:139-145), [plan.md](plan.md:237-244).

- Performance and scale
  - Risk: In-memory dataset handling; large Excel uploads; chart computations on full datasets.
  - Confirmations: API pagination for analytics, caching strategy, vector index persistence [plan.md](plan.md:216-221).

- SQL safety for history endpoint
  - Risk: f-string LIMIT in [get_analysis_history()](servicenow_analyzer.py:123-129).
  - To Confirm: Enforce integer validation at API layer; parameterize queries in repositories.


9) References

- Monolith constructs
  - [filter_by_departments()](servicenow_analyzer.py:60)
  - [init_database()](servicenow_analyzer.py:67)
  - [save_analysis()](servicenow_analyzer.py:103)
  - [get_analysis_history()](servicenow_analyzer.py:123)
  - [DatasetManager.__init__()](servicenow_analyzer.py:137)
  - [DatasetManager.add_dataset()](servicenow_analyzer.py:141)
  - [ChartGenerator.create_quality_distribution()](servicenow_analyzer.py:188)
  - [ChartGenerator.create_complexity_distribution()](servicenow_analyzer.py:197)
  - [ChartGenerator.create_department_volume()](servicenow_analyzer.py:206)
  - [ChartGenerator.create_reassignment_analysis()](servicenow_analyzer.py:215)
  - [ChartGenerator.create_product_distribution()](servicenow_analyzer.py:226)
  - [OnboardingAnalyzer.__init__()](servicenow_analyzer.py:255)
  - [OnboardingAnalyzer.prepare_ticket_context()](servicenow_analyzer.py:263)
  - [OnboardingAnalyzer.analyze_onboarding_opportunities()](servicenow_analyzer.py:321)
  - [OnboardingAnalyzer.batch_analyze()](servicenow_analyzer.py:381)
  - [main()](servicenow_analyzer.py:392)
  - [generate_report()](servicenow_analyzer.py:814)
  - [servicenow_analyzer.py](servicenow_analyzer.py)

- Plan and target architecture
  - Overview and goals [plan.md](plan.md:1-7)
  - Current monolith summary [plan.md](plan.md:8-16)
  - Target architecture and repository layout [plan.md](plan.md:18-23), [plan.md](plan.md:24-109)
  - Core data schema [plan.md](plan.md:111-120)
  - API specifications [plan.md](plan.md:121-133)
  - Processing pipelines [plan.md](plan.md:134-145)
  - Modular prompt management [plan.md](plan.md:146-151)
  - Sampling and context budget enhancements [plan.md](plan.md:152-157)
  - Prototype architecture diagram [plan.md](plan.md:160-183)
  - Milestones and acceptance [plan.md](plan.md:185-215)
  - Operational considerations [plan.md](plan.md:216-221)
  - Near-term refactors/mapping [plan.md](plan.md:222-233)
  - Technology choices [plan.md](plan.md:234-244)
  - Definition of Done [plan.md](plan.md:245-251)
  - References to current code [plan.md](plan.md:252-258)
  - [plan.md](plan.md)

Appendix: Additional Observations

- Module import side-effect: Database initialization runs at import time [servicenow_analyzer.py](servicenow_analyzer.py:131-133); in target system, move to startup hooks/migrations.
- Quality score helper: DatasetManager.get_comparison_stats references a private quality scoring helper (self._quality_score) [servicenow_analyzer.py](servicenow_analyzer.py:166-167); ensure equivalent metric function exists in Engine analytics.

End of Reference Behavior Review. This document enumerates preserved semantics, improvements aligned with [plan.md](plan.md:18-23), defines data contracts and invariants, and lists confirmations needed to finalize module boundaries and tests for the next Architecture & Design Spec.