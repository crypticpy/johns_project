from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.orm import Session

from ai.llm.factory import select_analyzer
from ai.rerank.factory import select_reranker
from ai.rerank.interface import RerankError
from db.models import Analysis, Dataset, Ticket
from db.repositories.analyses_repo import AnalysesRepository
from db.repositories.audit_repo import AuditRepository
from db.repositories.clusters_repo import ClustersRepository
from db.repositories.datasets_repo import DatasetsRepository
from db.repositories.embeddings_repo import EmbeddingsRepository
from db.repositories.tickets_repo import TicketsRepository
from db.session import SessionLocal
from engine.analytics.clustering import hdbscan_cluster, kmeans_cluster, tfidf_top_terms
from engine.features.sampling import SamplingConfig, stratified_sample
from engine.ingest.loader import validate_and_load
from vector_store.faiss_index import FaissIndexAdapter, FaissIndexError

# Optional metrics (Prometheus) — mirror pattern in src/observability/metrics.py
try:
    from prometheus_client import Counter, Histogram  # type: ignore

    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover
    Counter = None  # type: ignore[assignment]
    Histogram = None  # type: ignore[assignment]
    _PROM_AVAILABLE = False

# Embeddings factory used by routers
try:
    from ai.embeddings.factory import resolve_defaults, select_embedder  # type: ignore
except Exception as _e:
    # Defer import error to adapter execution time for clearer error mapping
    resolve_defaults = None  # type: ignore[assignment]
    select_embedder = None  # type: ignore[assignment]

# Observability: tracing
try:
    from observability.tracing import tracer  # type: ignore
except Exception:

    def _noop_tracer():
        return None

    tracer = _noop_tracer  # type: ignore[assignment]


_LOGGER = logging.getLogger("sd_onboarding")


# ----------------------------
# Pydantic Models (IO Schemas)
# ----------------------------


class StrictModel(BaseModel):
    # Forbid extra fields and disable protected namespace warnings (e.g., 'model_name')
    model_config = ConfigDict(extra="forbid", str_min_length=0, protected_namespaces=())


# 1) tool.ingest.upload
class IngestUploadInput(StrictModel):
    file_path: str = Field(min_length=1)
    dataset_name: str | None = None


class IngestUploadOutput(StrictModel):
    dataset_id: int = Field(ge=1)
    name: str
    row_count: int = Field(ge=0)
    department_count: int = Field(ge=0)
    file_hash: str
    inserted_tickets: int = Field(ge=0)


# 2) tool.embed.run


class EmbedRunInput(StrictModel):
    dataset_id: int = Field(ge=1)
    backend: Literal["sentence-transformers", "builtin"] | None = Field(default=None)
    model_name: str | None = None
    batch_size: int | None = Field(default=None, ge=1)


class EmbedRunOutput(StrictModel):
    dataset_id: int = Field(ge=1)
    model_name: str
    backend: str
    vector_dim: int = Field(ge=0)
    embedded_count: int = Field(ge=0)
    indexed: bool


# 3) tool.search.nn
class SearchFilters(StrictModel):
    department: list[str] | None = None
    product: list[str] | None = None


class SearchNNInput(StrictModel):
    dataset_id: int = Field(ge=1)
    query_text: str = Field(min_length=1)
    k: int | None = Field(default=10, ge=1)
    filters: SearchFilters | None = None
    rerank: bool | None = False
    rerank_backend: str | None = None  # "builtin" | "cross-encoder"


class SearchNNItem(StrictModel):
    ticket_id: int = Field(ge=1)
    score: float
    department: str | None = None
    product: str | None = None
    summary: str | None = None


class SearchNNOutput(StrictModel):
    dataset_id: int = Field(ge=1)
    k: int = Field(ge=1)
    backend: str | None = None
    model_name: str
    rerank: bool
    rerank_backend: str | None = None
    results: list[SearchNNItem]


# 4) tool.cluster.run
class ClusterParams(StrictModel):
    n_clusters: int | None = Field(default=None, ge=2)
    min_cluster_size: int | None = Field(default=None, ge=2)
    min_samples: int | None = Field(default=None, ge=1)


class ClusterRunInput(StrictModel):
    dataset_id: int = Field(ge=1)
    algorithm: str  # "kmeans" | "hdbscan"
    params: ClusterParams | None = None
    model_name: str | None = None


class ClusterRunOutput(StrictModel):
    dataset_id: int = Field(ge=1)
    algorithm: str
    model_name: str
    run_id: int = Field(ge=1)
    silhouette: float | None = None
    cluster_counts: dict[int, int]


# 5) tool.analysis.run
class AnalysisRunInput(StrictModel):
    dataset_id: int = Field(ge=1)
    question: str = Field(min_length=1)
    prompt_version: str | None = "v1"
    analyzer_backend: str | None = None  # "openai" | "offline"
    max_tickets: int | None = 50
    token_budget: int | None = 2000
    compare_dataset_id: int | None = Field(default=None, ge=1)


class AnalysisRunOutput(StrictModel):
    analysis_id: int = Field(ge=1)
    dataset_id: int = Field(ge=1)
    prompt_version: str
    ticket_count: int = Field(ge=0)
    created_at: str | None = None  # ISO8601


# 6) tool.reports.get
class ReportsGetInput(StrictModel):
    dataset_id: int = Field(ge=1)


class ReportsGetOutput(StrictModel):
    dataset_id: int = Field(ge=1)
    report_markdown: str
    analysis_count: int = Field(ge=0)


# 7) tool.prompts.list/load/save
class PromptsListInput(StrictModel):
    pass


class PromptsListOutput(StrictModel):
    versions: list[str]


class PromptsLoadInput(StrictModel):
    version: str = Field(min_length=1)


class PromptsLoadOutput(StrictModel):
    version: str
    template: str
    metadata: dict[str, Any] | None = None


class PromptsSaveInput(StrictModel):
    version: str = Field(min_length=1)
    template: str = Field(min_length=1)
    metadata: dict[str, Any] | None = None


class PromptsSaveOutput(StrictModel):
    ok: bool


# 8) tool.history.list
class HistoryListInput(StrictModel):
    limit: int = Field(ge=1, le=500, default=50)
    offset: int = Field(ge=0, default=0)
    dataset_id: int | None = Field(default=None, ge=1)
    prompt_version: str | None = None
    date_from: str | None = None
    date_to: str | None = None


class HistoryItem(StrictModel):
    id: int = Field(ge=1)
    dataset_id: int = Field(ge=1)
    prompt_version: str
    question: str
    ticket_count: int = Field(ge=0)
    created_at: str | None = None


class HistoryListOutput(StrictModel):
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    total: int = Field(ge=0)
    items: list[HistoryItem]


# ---------------------------------
# Tool Spec, Context, Error Mapping
# ---------------------------------


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    roles_required: set[str] | None = None
    audit_sensitive: bool = False
    adapter: Callable[[dict[str, Any], ToolContext, Session], dict[str, Any]] | None = None


@dataclass
class ToolContext:
    subject: str
    roles: set[str]
    request_id: str
    token_budget: int | None = None
    step_limit: int | None = None
    dataset_id: int | None = None


def build_tool_context_from_claims(
    claims: dict[str, Any],
    *,
    request_id: str,
    token_budget: int | None = None,
    step_limit: int | None = None,
    dataset_id: int | None = None,
) -> ToolContext:
    """
    Build ToolContext from decoded JWT claims.

    Roles extraction supports:
      - roles: List[str] or comma-separated string
      - role: str
    """
    roles: set[str] = set()
    raw_roles = claims.get("roles")
    if isinstance(raw_roles, list):
        for r in raw_roles:
            if isinstance(r, str) and r.strip():
                roles.add(r.strip().lower())
    elif isinstance(raw_roles, str):
        for r in raw_roles.split(","):
            if r.strip():
                roles.add(r.strip().lower())
    single = claims.get("role")
    if isinstance(single, str) and single.strip():
        roles.add(single.strip().lower())

    subject = str(claims.get("sub") or claims.get("subject") or claims.get("user") or "anonymous")
    return ToolContext(
        subject=subject,
        roles=roles,
        request_id=str(request_id),
        token_budget=token_budget,
        step_limit=step_limit,
        dataset_id=dataset_id,
    )


# Metrics for tools
_TOOL_CALL_COUNT = (
    Counter("tool_call_count", "Total tool calls", labelnames=("tool_name",))
    if _PROM_AVAILABLE
    else None
)
_TOOL_LATENCY_SECONDS = (
    Histogram(
        "tool_latency_seconds",
        "Tool call latency in seconds",
        labelnames=("tool_name",),
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
    if _PROM_AVAILABLE
    else None
)
_RERANK_LATENCY_SECONDS = (
    Histogram(
        "rerank_latency_seconds",
        "Rerank latency seconds",
        labelnames=("backend",),
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    )
    if _PROM_AVAILABLE
    else None
)


def _env_flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _sanitize_args_summary(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    # Keep low-cardinality identifiers only
    out: dict[str, Any] = {"tool_name": tool_name}
    for key in ("dataset_id", "k", "rerank", "rerank_backend", "version", "algorithm"):
        v = args.get(key)
        if v is not None:
            out[key] = v if isinstance(v, (int, float, str, bool)) else str(v)
    return out


def _sanitize_result_summary(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"tool_name": tool_name}
    for key in ("dataset_id", "analysis_id", "run_id", "analysis_count"):
        v = result.get(key)
        if v is not None:
            out[key] = v
    if "results" in result and isinstance(result["results"], list):
        out["result_count"] = len(result["results"])
    return out


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    s = str(value).strip()
    try:
        if len(s) <= 10 and "-" in s and "T" not in s:
            return datetime.fromisoformat(s + "T00:00:00")
        return datetime.fromisoformat(s)
    except Exception as e:
        raise ValueError(f"Invalid ISO datetime: {value}") from e


def _map_exception_category(e: Exception) -> str:
    msg = str(e).lower()
    if isinstance(e, ValidationError):
        return "validation_error"
    if isinstance(e, FaissIndexError):
        if "does not exist" in msg:
            return "downstream_error"
        if "dimension mismatch" in msg:
            return "downstream_error"
        return "downstream_error"
    if isinstance(e, RerankError):
        # backend unavailable or offline
        return "tool_unavailable"
    if "hdbscan" in msg and "unavailable" in msg:
        return "tool_unavailable"
    return "downstream_error"


# ---------------
# Tool Adapters
# ---------------


def _adapter_ingest_upload(args: dict[str, Any], ctx: ToolContext, db: Session) -> dict[str, Any]:
    file_path = str(args["file_path"])
    dataset_name = str(args.get("dataset_name") or "") or None
    # Read bytes safely
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}") from e

    df, meta = validate_and_load(file_bytes, os.path.basename(file_path))
    file_hash: str = str(meta["file_hash"])
    row_count: int = int(meta["rows"])

    ds = DatasetsRepository.create_or_get(
        db,
        name=dataset_name or os.path.basename(file_path),
        file_hash=file_hash,
        row_count=row_count,
        department_count=0,
        metadata={"filename": os.path.basename(file_path)},
    )

    # Insert tickets only for new dataset (avoid duplication)
    existing_tickets = TicketsRepository.query_filtered(db, dataset_id=ds.id, limit=1)
    inserted = 0
    if not existing_tickets and row_count > 0:
        inserted = TicketsRepository.bulk_insert(db, dataset_id=ds.id, df=df)

    dept_count = DatasetsRepository.recompute_department_count(db, ds.id)
    return IngestUploadOutput(
        dataset_id=ds.id,
        name=ds.name,
        row_count=ds.row_count,
        department_count=dept_count,
        file_hash=ds.file_hash,
        inserted_tickets=inserted,
    ).model_dump()


def _adapter_embed_run(args: dict[str, Any], ctx: ToolContext, db: Session) -> dict[str, Any]:
    dataset_id = int(args["dataset_id"])
    backend_in = args.get("backend")
    model_name_in = args.get("model_name")
    batch_size_in = args.get("batch_size")

    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise ValueError(f"Dataset {dataset_id} not found")

    if select_embedder is None or resolve_defaults is None:
        raise RuntimeError("Embeddings factory unavailable")

    embedder = select_embedder(backend_in)
    default_model, default_batch = resolve_defaults()
    model_name = (model_name_in or default_model).strip()
    batch_size = int(batch_size_in or default_batch)

    ticket_ids, texts = EmbeddingsRepository.fetch_candidate_texts(db, dataset_id=dataset_id)
    if not ticket_ids or not texts or len(ticket_ids) != len(texts):
        return EmbedRunOutput(
            dataset_id=dataset_id,
            model_name=model_name,
            backend=(backend_in or "sentence-transformers"),
            vector_dim=0,
            embedded_count=0,
            indexed=False,
        ).model_dump()

    vectors = embedder.embed_texts(texts, model=model_name, batch_size=batch_size)
    if not vectors or len(vectors) != len(ticket_ids):
        raise RuntimeError("Embedding adapter returned invalid vector count")

    EmbeddingsRepository.upsert_for_dataset(
        db,
        dataset_id=dataset_id,
        model_name=model_name,
        records=list(zip(ticket_ids, vectors, strict=False)),
        batch_size=1000,
    )

    ids_all, vecs_all = EmbeddingsRepository.fetch_by_dataset(
        db, dataset_id=dataset_id, model_name=model_name
    )
    vector_dim: int = 0 if not vecs_all else len(vecs_all[0])

    index = FaissIndexAdapter()
    if ids_all and vecs_all:
        index.build_index(
            dataset_id=dataset_id, vectors=vecs_all, ids=ids_all, model_name=model_name
        )

    return EmbedRunOutput(
        dataset_id=dataset_id,
        model_name=model_name,
        backend=(backend_in or "sentence-transformers"),
        vector_dim=vector_dim,
        embedded_count=len(ticket_ids),
        indexed=bool(ids_all and vecs_all),
    ).model_dump()


def _adapter_search_nn(args: dict[str, Any], ctx: ToolContext, db: Session) -> dict[str, Any]:
    dataset_id = int(args["dataset_id"])
    q_text = str(args["query_text"]).strip()
    k = int(args.get("k") or 10)
    filters = args.get("filters") or {}
    departments: list[str] | None = filters.get("department") or None
    products: list[str] | None = filters.get("product") or None
    rerank_flag: bool = bool(args.get("rerank") or False)
    rerank_backend_in: str | None = args.get("rerank_backend") or None

    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise ValueError(f"Dataset {dataset_id} not found")

    index = FaissIndexAdapter()
    meta = index._read_meta(dataset_id)  # type: ignore[attr-defined]
    model_name_from_meta = None
    if isinstance(meta, dict):
        model_name_from_meta = str(meta.get("model_name") or "").strip() or None

    if select_embedder is None or resolve_defaults is None:
        raise RuntimeError("Embeddings factory unavailable")

    backend_choice = None
    if model_name_from_meta and model_name_from_meta.strip().lower().startswith("builtin-"):
        backend_choice = "builtin"
    embedder = select_embedder(backend_choice or None)

    default_model, _default_batch = resolve_defaults()
    model_name = model_name_from_meta or default_model

    start_embed = time.monotonic()
    vecs = embedder.embed_texts([q_text], model=model_name, batch_size=32)
    _embed_latency = max(0.0, time.monotonic() - start_embed)
    if not vecs or not isinstance(vecs[0], list) or len(vecs[0]) == 0:
        raise RuntimeError("Embedding adapter returned invalid vector")
    query_vec = vecs[0]

    start_search = time.monotonic()
    nn: list[tuple[int, float]] = index.search(dataset_id=dataset_id, vector=query_vec, k=k)
    _search_latency = max(0.0, time.monotonic() - start_search)
    # Optional metrics: VECTOR_SEARCH_LATENCY in metrics.py is global middleware; we emit tool-level only here.

    if not nn:
        return SearchNNOutput(
            dataset_id=dataset_id,
            k=k,
            backend=backend_choice or None,
            model_name=model_name,
            rerank=bool(rerank_flag),
            rerank_backend=rerank_backend_in,
            results=[],
        ).model_dump()

    ordered_ids = [tid for tid, _ in nn]
    score_map: dict[int, float] = {tid: float(score) for tid, score in nn}

    # Fetch matched tickets
    from sqlalchemy import Select, and_  # local import to avoid global symbol clash
    from sqlalchemy import select as sa_select

    stmt: Select = sa_select(Ticket).where(
        and_(Ticket.id.in_(ordered_ids), Ticket.dataset_id == dataset_id)
    )
    rows: list[Ticket] = list(db.execute(stmt).scalars().all())
    by_id: dict[int, Ticket] = {int(t.id): t for t in rows}

    # Optional rerank
    if rerank_flag:
        reranker = select_reranker(
            "builtin"
            if (rerank_backend_in and rerank_backend_in.strip().lower() in ("builtin", "lexical"))
            else (
                "cross-encoder"
                if rerank_backend_in
                and rerank_backend_in.strip().lower().replace("_", "-")
                in ("cross-encoder", "crossencoder")
                else None
            )  # type: ignore[arg-type]
        )
        candidates: list[tuple[int, str]] = []
        for tid in ordered_ids:
            t = by_id.get(int(tid))
            if t is None:
                continue
            text = (t.summary or "") or (t.normalized_text or "")
            candidates.append((int(tid), text))
        start_rerank = time.monotonic()
        reranked = reranker.rerank(q_text, candidates)
        rr_latency = max(0.0, time.monotonic() - start_rerank)
        if _PROM_AVAILABLE and _RERANK_LATENCY_SECONDS is not None:
            backend_label = (
                "cross-encoder" if type(reranker).__name__ == "CrossEncoderReranker" else "builtin"
            )
            # type: ignore[union-attr]
            _RERANK_LATENCY_SECONDS.labels(backend=backend_label).observe(
                rr_latency
            )  # pragma: no cover
        ordered_ids = [tid for tid, _ in reranked]
        score_map = {tid: float(score) for tid, score in reranked}

    dept_set = set([d for d in (departments or []) if isinstance(d, str)])
    prod_set = set([p for p in (products or []) if isinstance(p, str)])

    results: list[SearchNNItem] = []
    for tid in ordered_ids:
        t = by_id.get(int(tid))
        if t is None:
            continue
        if dept_set and (t.department is None or t.department not in dept_set):
            continue
        if prod_set and (t.product is None or t.product not in prod_set):
            continue
        results.append(
            SearchNNItem(
                ticket_id=int(t.id),
                score=float(score_map.get(int(t.id), 0.0)),
                department=t.department,
                product=t.product,
                summary=t.summary,
            )
        )
        if len(results) >= k:
            break

    return SearchNNOutput(
        dataset_id=dataset_id,
        k=k,
        backend=backend_choice or None,
        model_name=model_name,
        rerank=bool(rerank_flag),
        rerank_backend=(
            "builtin"
            if rerank_backend_in and rerank_backend_in.strip().lower() in ("builtin", "lexical")
            else (
                "cross-encoder"
                if rerank_backend_in
                and rerank_backend_in.strip().lower().replace("_", "-")
                in ("cross-encoder", "crossencoder")
                else None
            )
        ),
        results=[r for r in results],
    ).model_dump()


def _adapter_cluster_run(args: dict[str, Any], ctx: ToolContext, db: Session) -> dict[str, Any]:
    dataset_id = int(args["dataset_id"])
    algorithm = str(args["algorithm"]).strip().lower()
    params_in: dict[str, Any] = dict(args.get("params") or {})
    model_name_in = args.get("model_name")

    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise ValueError(f"Dataset {dataset_id} not found")

    index = FaissIndexAdapter()
    model_name = None
    if isinstance(model_name_in, str) and model_name_in.strip():
        model_name = model_name_in.strip()
    else:
        meta = index._read_meta(dataset_id)  # type: ignore[attr-defined]
        if isinstance(meta, dict):
            mn = str(meta.get("model_name") or "").strip()
            model_name = mn or None
    if not model_name:
        raise ValueError("Embeddings model_name could not be resolved for dataset")

    if not EmbeddingsRepository.exists_for_dataset(
        db, dataset_id=dataset_id, model_name=model_name
    ):
        raise ValueError(f"Embeddings for dataset {dataset_id} and model '{model_name}' not found")

    ids_all, vecs_all = EmbeddingsRepository.fetch_by_dataset(
        db, dataset_id=dataset_id, model_name=model_name
    )
    if not ids_all or not vecs_all or len(ids_all) != len(vecs_all):
        raise RuntimeError("No embeddings vectors available for clustering")

    ticket_ids_texts, candidate_texts = EmbeddingsRepository.fetch_candidate_texts(
        db, dataset_id=dataset_id, limit=250_000
    )
    text_map: dict[int, str] = {
        int(tid): str(text or "").strip()
        for tid, text in zip(ticket_ids_texts, candidate_texts, strict=False)
    }
    texts_aligned: list[str] = [text_map.get(int(tid), "") for tid in ids_all]

    # Engine clustering
    if algorithm == "kmeans":
        n_clusters_val = params_in.get("n_clusters", None)
        if n_clusters_val is None:
            raise ValueError("params.n_clusters is required for kmeans")
        n_clusters_int = int(n_clusters_val)
        result = kmeans_cluster(vectors=vecs_all, n_clusters=n_clusters_int, random_state=42)
    elif algorithm == "hdbscan":
        min_cluster_size = int(params_in.get("min_cluster_size", 5))
        min_samples = int(params_in.get("min_samples", 5))
        try:
            result = hdbscan_cluster(
                vectors=vecs_all, min_cluster_size=min_cluster_size, min_samples=min_samples
            )
        except RuntimeError as e:
            # Optional dependency unavailable
            raise RuntimeError(f"HDBSCAN unavailable: {e}") from e
    else:
        raise ValueError("algorithm must be 'kmeans' or 'hdbscan'")

    assignments: list[int] = list(result.get("assignments") or [])
    silhouette: float | None = (
        result.get("silhouette") if isinstance(result.get("silhouette"), (float, int)) else None
    )
    if len(assignments) != len(ids_all):
        raise RuntimeError("Engine returned invalid assignments length")

    run_id = ClustersRepository.create_run(
        db,
        dataset_id=dataset_id,
        model_name=model_name,
        algorithm=algorithm,
        params={
            "algorithm": algorithm,
            **(
                {"n_clusters": int(params_in.get("n_clusters"))}
                if algorithm == "kmeans" and params_in.get("n_clusters") is not None
                else {}
            ),
            **(
                {"min_cluster_size": int(params_in.get("min_cluster_size", 5))}
                if algorithm == "hdbscan"
                else {}
            ),
            **(
                {"min_samples": int(params_in.get("min_samples", 5))}
                if algorithm == "hdbscan"
                else {}
            ),
        },
    )
    tuples: list[tuple[int, int]] = [
        (int(tid), int(cid)) for tid, cid in zip(ids_all, assignments, strict=False)
    ]
    ClustersRepository.store_assignments(db, run_id=run_id, assignments=tuples, batch_size=1000)
    ClustersRepository.store_metrics(db, run_id=run_id, metrics={"silhouette": silhouette})

    top_terms = tfidf_top_terms(texts=texts_aligned, assignments=assignments, top_k=10)
    ClustersRepository.store_top_terms(db, run_id=run_id, top_terms=top_terms, batch_size=1000)

    summary = ClustersRepository.fetch_run_summary(db, run_id=run_id)

    # Audit (sensitive)
    try:
        AuditRepository.record(
            db,
            subject=ctx.subject,
            action="cluster.run",
            resource=f"dataset:{dataset_id}",
            metadata={
                "run_id": int(run_id),
                "algorithm": algorithm,
                "model_name": str(model_name),
            },
        )
    except Exception:
        pass

    return ClusterRunOutput(
        dataset_id=dataset_id,
        algorithm=algorithm,
        model_name=model_name,
        run_id=int(run_id),
        silhouette=summary.get("silhouette"),
        cluster_counts=summary.get("cluster_counts", {}),
    ).model_dump()


def _adapter_analysis_run(args: dict[str, Any], ctx: ToolContext, db: Session) -> dict[str, Any]:
    dataset_id = int(args["dataset_id"])
    question = str(args["question"]).strip()
    prompt_version = str(args.get("prompt_version") or "v1").strip() or "v1"
    analyzer_backend = args.get("analyzer_backend")
    max_tickets = int(args.get("max_tickets") or 50)
    token_budget = int(args.get("token_budget") or 2000)
    compare_dataset_id = args.get("compare_dataset_id")
    comparison_mode = compare_dataset_id is not None

    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise ValueError(f"Dataset {dataset_id} not found")
    if comparison_mode:
        ds2 = db.get(Dataset, int(compare_dataset_id))
        if ds2 is None:
            raise ValueError(f"Comparison dataset {compare_dataset_id} not found")

    tickets: list[Ticket] = TicketsRepository.query_filtered(
        db, dataset_id=dataset_id, limit=100_000, offset=0
    )
    # Build canonical DataFrame as in router
    import pandas as pd

    def _tickets_to_df_local(tks: list[Ticket]) -> pd.DataFrame:
        if not tks:
            return pd.DataFrame(
                columns=[
                    "id",
                    "Department",
                    "extract_product",
                    "ticket_quality",
                    "resolution_complexity",
                    "Reassignment group count tracking_index",
                    "summarize_ticket",
                ]
            )
        records: list[dict[str, Any]] = []
        for t in tks:
            records.append(
                {
                    "id": int(t.id),
                    "Department": t.department,
                    "extract_product": t.product,
                    "ticket_quality": t.quality,
                    "resolution_complexity": t.complexity,
                    "Reassignment group count tracking_index": t.reassignment_count,
                    "summarize_ticket": t.summary,
                }
            )
        return pd.DataFrame.from_records(records)

    df = _tickets_to_df_local(tickets)
    cfg = SamplingConfig(max_tickets=int(max_tickets or 0), token_budget=int(token_budget or 0))
    sample = stratified_sample(df, cfg)

    context_text = str(sample.get("context_text") or "").strip() or "# Analysis Context\n(no data)"
    # Comparison block (compact)
    comparison_metrics: dict[str, Any] = {}
    if comparison_mode:
        # Reuse router behavior
        def _build_comparison_section_local(
            session: Session, compare_id: int
        ) -> tuple[str, dict[str, Any]]:
            ds2 = session.get(Dataset, int(compare_id))
            if ds2 is None:
                raise ValueError(f"Comparison dataset {compare_id} not found")
            tickets2: list[Ticket] = TicketsRepository.query_filtered(
                session, dataset_id=int(compare_id), limit=100_000, offset=0
            )
            df2 = _tickets_to_df_local(tickets2)
            from engine.analytics.metrics import (
                compute_complexity_distribution,
                compute_department_volume,
                compute_quality_distribution,
            )

            top_depts2 = compute_department_volume(df2, top_n=5)
            quality2 = compute_quality_distribution(df2)
            complexity2 = compute_complexity_distribution(df2)
            lines: list[str] = []
            lines.append(f"## Comparison Dataset Summary (dataset_id={compare_id})")
            total2 = int(len(df2))
            lines.append(f"Total tickets: {total2}")
            if top_depts2:
                dep_pairs = [f"{d}({c})" for d, c in top_depts2]
                lines.append("Top Departments: " + ", ".join(dep_pairs))
            if quality2:
                q_pairs = [f"{k}({v})" for k, v in sorted(quality2.items())]
                lines.append("Quality: " + ", ".join(q_pairs))
            if complexity2:
                c_pairs = [f"{k}({v})" for k, v in sorted(complexity2.items())]
                lines.append("Complexity: " + ", ".join(c_pairs))
            metrics_summary = {
                "total_rows": total2,
                "top_departments": top_depts2,
                "quality": quality2,
                "complexity": complexity2,
            }
            return ("\n".join(lines).strip(), metrics_summary)

        compare_md, compare_summary = _build_comparison_section_local(db, int(compare_dataset_id))
        context_text = f"{context_text}\n\n{compare_md}"
        comparison_metrics = {"comparison": compare_summary}

    try:
        analyzer = select_analyzer(
            str(analyzer_backend).strip().lower() if analyzer_backend else None
        )  # type: ignore[arg-type]
    except Exception as e:
        raise RuntimeError(str(e)) from e

    result_md = analyzer.analyze(
        context=context_text,
        question=question,
        prompt_version=prompt_version,
        comparison_mode=bool(comparison_mode),
    )

    sampled_ids: list[int] = list(sample.get("sampled_ids") or [])
    segments: list[dict[str, Any]] = list(sample.get("segments") or [])
    summary: dict[str, Any] = dict(sample.get("summary") or {})

    def _derive_departments_from_segments_local(segs: list[dict[str, Any]]) -> list[str]:
        depts: set[str] = set()
        for seg in segs or []:
            keymap = seg.get("key") or {}
            dep = keymap.get("Department")
            if isinstance(dep, str) and dep and dep != "(missing)":
                depts.add(dep)
        return sorted(depts)

    departments_used = _derive_departments_from_segments_local(segments)
    metrics: dict[str, Any] = {
        "sampling_summary": summary,
        "segments": segments,
        # Estimated tokens ~= chars/4
        "estimated_prompt_tokens": max(0, int(len(context_text) / 4)),
    }
    if comparison_metrics:
        metrics.update(comparison_metrics)
    filters: dict[str, Any] = {}
    if departments_used:
        filters["departments"] = departments_used

    analysis_id = AnalysesRepository.save_analysis(
        db,
        dataset_id=dataset_id,
        prompt_version=prompt_version,
        question=question,
        result_markdown=result_md,
        ticket_count=len(sampled_ids),
        metrics=metrics,
        filters=filters or None,
    )

    # Audit
    try:
        AuditRepository.record(
            db,
            subject=ctx.subject,
            action="analysis.run",
            resource=f"dataset:{dataset_id}",
            metadata={
                "analysis_id": int(analysis_id),
                "prompt_version": str(prompt_version),
            },
        )
    except Exception:
        pass

    created_at_iso: str | None = None
    try:
        row = db.get(Analysis, int(analysis_id))
        if row and isinstance(row.created_at, datetime):
            created_at_iso = row.created_at.isoformat()
    except Exception:
        created_at_iso = None

    return AnalysisRunOutput(
        analysis_id=int(analysis_id),
        dataset_id=int(dataset_id),
        prompt_version=str(prompt_version),
        ticket_count=int(len(sampled_ids)),
        created_at=created_at_iso,
    ).model_dump()


def _adapter_reports_get(args: dict[str, Any], ctx: ToolContext, db: Session) -> dict[str, Any]:
    dataset_id = int(args["dataset_id"])
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise ValueError(f"Dataset {dataset_id} not found")

    analyses = AnalysesRepository.list_analyses(
        db,
        limit=10,
        offset=0,
        dataset_id=dataset_id,
        prompt_version=None,
        date_from=None,
        date_to=None,
    )
    analysis_count = len(analyses)

    # Build markdown
    from engine.analytics.metrics import (
        compute_complexity_distribution,
        compute_department_volume,
        compute_product_distribution,
        compute_quality_distribution,
        compute_reassignment_distribution,
    )

    tickets: list[Ticket] = TicketsRepository.query_filtered(
        db, dataset_id=dataset_id, limit=100_000, offset=0
    )

    import pandas as pd

    def _tickets_to_df_local(tks: list[Ticket]) -> pd.DataFrame:
        if not tks:
            return pd.DataFrame(
                columns=[
                    "Department",
                    "extract_product",
                    "ticket_quality",
                    "resolution_complexity",
                    "Reassignment group count tracking_index",
                    "summarize_ticket",
                ]
            )
        records: list[dict[str, Any]] = []
        for t in tks:
            records.append(
                {
                    "Department": t.department,
                    "extract_product": t.product,
                    "ticket_quality": t.quality,
                    "resolution_complexity": t.complexity,
                    "Reassignment group count tracking_index": t.reassignment_count,
                    "summarize_ticket": t.summary,
                }
            )
        return pd.DataFrame.from_records(records)

    df = _tickets_to_df_local(tickets)
    lines: list[str] = []
    lines.append(f"# Dataset Report (dataset_id={dataset_id})\n")
    lines.append("## Recent Analyses")
    if analyses:
        for a in analyses:
            title = (a.question or "").strip()
            pv = (a.prompt_version or "").strip()
            created = a.created_at.isoformat() if getattr(a, "created_at", None) else "n/a"
            lines.append(f"- [{created}] prompt={pv} tickets={int(a.ticket_count or 0)} — {title}")
    else:
        lines.append("- No analyses available.")

    # Snapshot
    lines.append("\n## Current Analytics Snapshot")
    lines.append(f"Total Tickets: {int(len(df))}")
    dept_vol: list[tuple[str, int]] = compute_department_volume(df, top_n=10)
    lines.append("### Top Departments by Volume")
    if dept_vol:
        for dep, cnt in dept_vol:
            lines.append(f"- {dep}: {cnt}")
    else:
        lines.append("- (none)")
    quality = compute_quality_distribution(df)
    lines.append("### Ticket Quality Distribution")
    if quality:
        for k, v in sorted(quality.items()):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- (none)")
    complexity = compute_complexity_distribution(df)
    lines.append("### Resolution Complexity Distribution")
    if complexity:
        for k, v in sorted(complexity.items()):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- (none)")
    reassignment = compute_reassignment_distribution(df)
    lines.append("### Reassignment Count Distribution")
    if reassignment:
        for k in sorted(reassignment.keys()):
            lines.append(f"- {int(k)}: {int(reassignment[k])}")
    else:
        lines.append("- (none)")
    product = compute_product_distribution(df)
    lines.append("### Top Products by Volume")
    if product:
        for prod, cnt in product:
            lines.append(f"- {prod}: {cnt}")
    else:
        lines.append("- (none)")

    report_md = "\n".join(lines).strip()

    return ReportsGetOutput(
        dataset_id=int(dataset_id),
        report_markdown=report_md,
        analysis_count=int(analysis_count),
    ).model_dump()


def _adapter_prompts_list(args: dict[str, Any], ctx: ToolContext, db: Session) -> dict[str, Any]:
    from ai.llm.prompts.store import list_versions

    return PromptsListOutput(versions=list_versions()).model_dump()


def _adapter_prompts_load(args: dict[str, Any], ctx: ToolContext, db: Session) -> dict[str, Any]:
    from ai.llm.prompts.store import META_SUFFIX, TEMPLATES_DIR, load_template

    version = str(args["version"])
    content = load_template(version)
    # Load metadata sidecar if present
    import json
    from pathlib import Path

    meta_path = Path(TEMPLATES_DIR) / f"{version}{META_SUFFIX}"
    metadata: dict[str, Any] | None = None
    try:
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        metadata = None
    return PromptsLoadOutput(version=version, template=content, metadata=metadata).model_dump()


def _adapter_prompts_save(args: dict[str, Any], ctx: ToolContext, db: Session) -> dict[str, Any]:
    from ai.llm.prompts.store import save_template

    version = str(args["version"])
    template_body = str(args["template"])
    metadata_in: dict[str, Any] = dict(args.get("metadata") or {})
    metadata_in["template"] = template_body
    result = save_template(version=version, metadata=metadata_in)
    ok = bool(result.get("meta_written"))
    # Audit
    try:
        AuditRepository.record(
            db,
            subject=ctx.subject,
            action="prompts.save",
            resource=f"prompt:{version}",
            metadata={"version": version},
        )
    except Exception:
        pass
    return PromptsSaveOutput(ok=ok).model_dump()


def _adapter_history_list(args: dict[str, Any], ctx: ToolContext, db: Session) -> dict[str, Any]:
    limit = int(args.get("limit") or 50)
    offset = int(args.get("offset") or 0)
    dataset_id = args.get("dataset_id")
    prompt_version = args.get("prompt_version")
    date_from_raw = args.get("date_from")
    date_to_raw = args.get("date_to")
    df = _parse_iso_datetime(date_from_raw) if date_from_raw else None
    dt = _parse_iso_datetime(date_to_raw) if date_to_raw else None

    rows = AnalysesRepository.list_analyses(
        db,
        limit=limit,
        offset=offset,
        dataset_id=int(dataset_id) if dataset_id is not None else None,
        prompt_version=str(prompt_version) if prompt_version else None,
        date_from=df,
        date_to=dt,
    )
    total = AnalysesRepository.count_analyses(
        db,
        dataset_id=int(dataset_id) if dataset_id is not None else None,
        prompt_version=str(prompt_version) if prompt_version else None,
        date_from=df,
        date_to=dt,
    )

    items: list[HistoryItem] = []
    for r in rows:
        items.append(
            HistoryItem(
                id=int(r.id),
                dataset_id=int(r.dataset_id),
                prompt_version=r.prompt_version,
                question=r.question,
                ticket_count=int(r.ticket_count or 0),
                created_at=r.created_at.isoformat() if getattr(r, "created_at", None) else None,
            )
        )
    return HistoryListOutput(
        limit=int(limit), offset=int(offset), total=int(total), items=items
    ).model_dump()


# ----------------
# Tool Registry
# ----------------


class ToolRegistry:
    """
    Tool Registry with validation, RBAC, audit, and observability.

    Methods:
      - register_tool(spec)
      - get_spec(name)
      - validate_args(name, args)
      - execute(name, args, context)
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._rbac_enabled = _env_flag("APP_ENABLE_RBAC", True)
        self._tracing_enabled = _env_flag("APP_ENABLE_TRACING", False)
        self._register_all()

    def register_tool(self, spec: ToolSpec) -> None:
        if not spec.name or not spec.adapter:
            raise ValueError("ToolSpec requires name and adapter")
        self._tools[spec.name] = spec

    def get_spec(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def validate_args(self, name: str, args: dict[str, Any]) -> BaseModel:
        spec = self.get_spec(name)
        if spec is None:
            raise ValueError(f"Unknown tool: {name}")
        # Strict validation
        return spec.input_model.model_validate(args)

    def execute(self, name: str, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        """
        Execute a tool with:
          1) name whitelist
          2) input validation
          3) RBAC check
          4) adapter execution
          5) output validation
          6) audit emission for sensitive tools
          7) observability: logging, metrics, tracing
        """
        spec = self.get_spec(name)
        if spec is None:
            return {
                "error": {
                    "category": "validation_error",
                    "message": "Unknown tool name",
                    "details": {"tool_name": name},
                }
            }

        # Validate args
        try:
            validated_input = spec.input_model.model_validate(args)
        except ValidationError as e:
            return {
                "error": {
                    "category": "validation_error",
                    "message": "Invalid input payload",
                    "details": {"tool_name": name, "hint": str(e)},
                }
            }

        # RBAC
        if self._rbac_enabled and spec.roles_required and len(spec.roles_required) > 0:
            caller_roles = {r.strip().lower() for r in (context.roles or set())}
            required_roles = {r.strip().lower() for r in spec.roles_required}
            if not caller_roles.intersection(required_roles):
                return {
                    "error": {
                        "category": "rbac_denied",
                        "message": "Insufficient role",
                        "details": {
                            "tool_name": name,
                            "hint": f"required any of: {sorted(required_roles)}",
                        },
                    }
                }

        # Observability: tracing span
        span_ctx = None
        if self._tracing_enabled:
            tr = tracer()
            if tr:
                span_ctx = tr.start_as_current_span("tool.call")  # type: ignore[attr-defined]
        start = time.monotonic()
        if _PROM_AVAILABLE and _TOOL_CALL_COUNT is not None:
            # type: ignore[union-attr]
            _TOOL_CALL_COUNT.labels(tool_name=name).inc()  # pragma: no cover

        db = SessionLocal()
        try:
            # Adapter execution
            result_raw = spec.adapter(validated_input.model_dump(), context, db)  # type: ignore[arg-type]
            # Output validation
            try:
                out_model = spec.output_model.model_validate(result_raw)
            except ValidationError as e:
                return {
                    "error": {
                        "category": "validation_error",
                        "message": "Adapter returned invalid output",
                        "details": {"tool_name": name, "hint": str(e)},
                    }
                }

            result = out_model.model_dump()
            duration = max(0.0, time.monotonic() - start)
            if _PROM_AVAILABLE and _TOOL_LATENCY_SECONDS is not None:
                # type: ignore[union-attr]
                _TOOL_LATENCY_SECONDS.labels(tool_name=name).observe(duration)  # pragma: no cover

            # Logging
            _LOGGER.info(
                "tool.execute: name=%s duration=%.4fs args=%s result=%s",
                name,
                duration,
                _sanitize_args_summary(name, validated_input.model_dump()),
                _sanitize_result_summary(name, result),
            )

            # Tracing attrs
            if span_ctx:
                try:
                    span = span_ctx.__enter__()  # type: ignore[attr-defined]
                    # type: ignore[attr-defined]
                    span.set_attribute("tool_name", name)
                    # type: ignore[attr-defined]
                    span.set_attribute(
                        "args_summary",
                        str(_sanitize_args_summary(name, validated_input.model_dump())),
                    )
                    # type: ignore[attr-defined]
                    span.set_attribute(
                        "result_summary", str(_sanitize_result_summary(name, result))
                    )
                    span_ctx.__exit__(None, None, None)  # type: ignore[attr-defined]
                except Exception:
                    pass

            return result
        except Exception as e:
            duration = max(0.0, time.monotonic() - start)
            if _PROM_AVAILABLE and _TOOL_LATENCY_SECONDS is not None:
                # type: ignore[union-attr]
                _TOOL_LATENCY_SECONDS.labels(tool_name=name).observe(duration)  # pragma: no cover
            _LOGGER.error(
                "tool.execute.error: name=%s duration=%.4fs error=%s", name, duration, str(e)
            )
            category = _map_exception_category(e)
            return {
                "error": {
                    "category": category,
                    "message": "Tool execution failed",
                    "details": {"tool_name": name, "hint": str(e)},
                }
            }
        finally:
            try:
                db.close()
            except Exception:
                pass

    def _register_all(self) -> None:
        # Role map per spec
        default_roles = {"viewer", "analyst", "admin"}
        roles_map: dict[str, set[str] | None] = {
            "analysis.run": {"analyst", "admin"},
            "history.list": {"viewer", "admin"},
            "cluster.run": {"analyst", "admin"},
            "prompts.save": {"admin"},
            # defaults for others
        }

        # 1) ingest.upload
        self.register_tool(
            ToolSpec(
                name="ingest.upload",
                description="Upload CSV/Excel and register dataset",
                input_model=IngestUploadInput,
                output_model=IngestUploadOutput,
                roles_required=default_roles,
                audit_sensitive=False,
                adapter=_adapter_ingest_upload,
            )
        )
        # 2) embed.run
        self.register_tool(
            ToolSpec(
                name="embed.run",
                description="Generate embeddings for a dataset and persist/index",
                input_model=EmbedRunInput,
                output_model=EmbedRunOutput,
                roles_required=default_roles,
                audit_sensitive=False,
                adapter=_adapter_embed_run,
            )
        )
        # 3) search.nn
        self.register_tool(
            ToolSpec(
                name="search.nn",
                description="Vector search on a dataset with optional rerank",
                input_model=SearchNNInput,
                output_model=SearchNNOutput,
                roles_required=default_roles,
                audit_sensitive=False,
                adapter=_adapter_search_nn,
            )
        )
        # 4) cluster.run
        self.register_tool(
            ToolSpec(
                name="cluster.run",
                description="Cluster embeddings and compute metrics/top terms",
                input_model=ClusterRunInput,
                output_model=ClusterRunOutput,
                roles_required=roles_map.get("cluster.run"),
                audit_sensitive=True,
                adapter=_adapter_cluster_run,
            )
        )
        # 5) analysis.run
        self.register_tool(
            ToolSpec(
                name="analysis.run",
                description="Stratified sampling plus LLM analysis; persist results",
                input_model=AnalysisRunInput,
                output_model=AnalysisRunOutput,
                roles_required=roles_map.get("analysis.run"),
                audit_sensitive=True,
                adapter=_adapter_analysis_run,
            )
        )
        # 6) reports.get
        self.register_tool(
            ToolSpec(
                name="reports.get",
                description="Assemble markdown report from recent analyses and analytics snapshot",
                input_model=ReportsGetInput,
                output_model=ReportsGetOutput,
                roles_required=default_roles,
                audit_sensitive=False,
                adapter=_adapter_reports_get,
            )
        )
        # 7) prompts.* (list/load/save)
        self.register_tool(
            ToolSpec(
                name="prompts.list",
                description="List available prompt template versions",
                input_model=PromptsListInput,
                output_model=PromptsListOutput,
                roles_required=default_roles,
                audit_sensitive=False,
                adapter=_adapter_prompts_list,
            )
        )
        self.register_tool(
            ToolSpec(
                name="prompts.load",
                description="Load a prompt template by version",
                input_model=PromptsLoadInput,
                output_model=PromptsLoadOutput,
                roles_required=default_roles,
                audit_sensitive=False,
                adapter=_adapter_prompts_load,
            )
        )
        self.register_tool(
            ToolSpec(
                name="prompts.save",
                description="Save or update a prompt template and metadata",
                input_model=PromptsSaveInput,
                output_model=PromptsSaveOutput,
                roles_required=roles_map.get("prompts.save"),
                audit_sensitive=True,
                adapter=_adapter_prompts_save,
            )
        )
        # 8) history.list
        self.register_tool(
            ToolSpec(
                name="history.list",
                description="Paginated analyses history",
                input_model=HistoryListInput,
                output_model=HistoryListOutput,
                roles_required=roles_map.get("history.list"),
                audit_sensitive=False,
                adapter=_adapter_history_list,
            )
        )


__all__ = [
    "ToolSpec",
    "ToolContext",
    "ToolRegistry",
    "build_tool_context_from_claims",
]
