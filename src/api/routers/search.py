from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import Select, and_, select
from sqlalchemy.orm import Session

from ai.embeddings.factory import resolve_defaults, select_embedder
from ai.rerank.factory import select_reranker
from ai.rerank.interface import RerankError
from db.models import Dataset, Ticket
from db.session import get_session
from vector_store.faiss_index import FaissIndexAdapter, FaissIndexError

router = APIRouter(prefix="/search", tags=["search"])


def _determine_backend_from_model(model_name: str | None) -> Optional[str]:
    """
    Heuristic: choose embeddings backend based on the model name recorded in FAISS metadata.
    - builtin-* -> 'builtin'
    - otherwise -> None (use factory default/environment)
    """
    if not model_name:
        return None
    m = model_name.strip().lower()
    if m.startswith("builtin-"):
        return "builtin"
    return None


@router.get("/nn", response_class=JSONResponse)
async def knn_search(request: Request, db: Session = Depends(get_session)) -> JSONResponse:
    """
    kNN semantic search over a dataset.

    Query params:
      - dataset_id: int (required)
      - q: str (required) query text
      - k: int (default=10)
      - department: optional repeated param for department filter(s)
      - product: optional repeated param for product filter(s)
    """
    # Parse and validate query params defensively
    qp = request.query_params

    dataset_id_val = qp.get("dataset_id")
    if dataset_id_val is None:
        raise HTTPException(status_code=400, detail="dataset_id is required")
    try:
        dataset_id = int(dataset_id_val)
    except Exception:
        raise HTTPException(status_code=400, detail="dataset_id must be a positive integer")
    if dataset_id <= 0:
        raise HTTPException(status_code=400, detail="dataset_id must be a positive integer")

    q = (qp.get("q") or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="q is required")

    k_val = qp.get("k", "10")
    try:
        k = int(k_val)
    except Exception:
        raise HTTPException(status_code=400, detail="k must be a positive integer")
    if k <= 0:
        raise HTTPException(status_code=400, detail="k must be a positive integer")

    departments: Optional[List[str]] = list(qp.getlist("department")) or None
    products: Optional[List[str]] = list(qp.getlist("product")) or None

    # Optional rerank flags
    rerank_flag_raw = qp.get("rerank", "false")
    rerank_flag: bool = str(rerank_flag_raw).strip().lower() in ("1", "true", "yes", "on")
    rerank_backend_in = qp.get("rerank_backend")
    rerank_backend_norm: Optional[str] = None
    if rerank_backend_in:
        rb = rerank_backend_in.strip().lower()
        if rb in ("builtin", "lexical"):
            rerank_backend_norm = "builtin"
        elif rb in ("cross-encoder", "crossencoder", "cross_encoder"):
            rerank_backend_norm = "cross-encoder"
        else:
            rerank_backend_norm = None

    # Validate dataset
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    # Resolve model_name from FAISS metadata for dimensional compatibility
    index = FaissIndexAdapter()
    meta = index._read_meta(dataset_id)  # type: ignore[attr-defined]
    model_name_from_meta = None
    if isinstance(meta, dict):
        model_name_from_meta = str(meta.get("model_name") or "").strip() or None

    # Choose backend: prefer one inferred from model_name; else resolve from environment
    backend_choice = _determine_backend_from_model(model_name_from_meta)
    embedder = select_embedder(backend_choice or None)

    # Determine model_name for embedding the query
    default_model, _default_batch = resolve_defaults()
    model_name = model_name_from_meta or default_model

    # Embed the query text
    try:
        vecs = embedder.embed_texts([q or ""], model=model_name, batch_size=32)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding backend error: {e}")
    if not vecs or not isinstance(vecs[0], list) or len(vecs[0]) == 0:
        raise HTTPException(status_code=500, detail="Embedding adapter returned invalid vector")

    query_vec = vecs[0]

    # Run FAISS search
    try:
        nn: List[Tuple[int, float]] = index.search(dataset_id=dataset_id, vector=query_vec, k=int(k))
    except FaissIndexError as e:
        # Missing index or dim mismatch -> 404/409 depending on message
        msg = str(e)
        if "does not exist" in msg:
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=409, detail=f"FAISS index error: {msg}")

    if not nn:
        payload = {
            "dataset_id": dataset_id,
            "k": int(k),
            "backend": backend_choice or (None),
            "model_name": model_name,
            "rerank": bool(rerank_flag),
            "rerank_backend": rerank_backend_norm,
            "results": [],
        }
        return JSONResponse(payload)

    # Preserve order from FAISS by building id -> score map
    ordered_ids = [tid for tid, _ in nn]
    score_map: Dict[int, float] = {tid: float(score) for tid, score in nn}

    # Fetch matched tickets (single query)
    stmt: Select = select(Ticket).where(and_(Ticket.id.in_(ordered_ids), Ticket.dataset_id == dataset_id))
    rows: List[Ticket] = list(db.execute(stmt).scalars().all())
    by_id: Dict[int, Ticket] = {int(t.id): t for t in rows}

    # Optional rerank of top-k candidates
    if rerank_flag:
        try:
            reranker = select_reranker(rerank_backend_norm)  # type: ignore[arg-type]
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Rerank backend selection failed: {e}")

        candidates: List[Tuple[int, str]] = []
        for tid in ordered_ids:
            t = by_id.get(int(tid))
            if t is None:
                continue
            # Use summary primarily; fallback to normalized_text for robustness
            text = (t.summary or "") or (t.normalized_text or "")
            candidates.append((int(tid), text))

        try:
            reranked = reranker.rerank(q, candidates)
        except RerankError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Rerank error: {e}")

        # Replace ordering and score map with reranked results
        ordered_ids = [tid for tid, _ in reranked]
        score_map = {tid: float(score) for tid, score in reranked}

    # Apply optional filters to results while preserving FAISS order
    dept_set = set([d for d in (departments or []) if d is not None])
    prod_set = set([p for p in (products or []) if p is not None])

    results: List[Dict[str, Any]] = []
    for tid in ordered_ids:
        t = by_id.get(int(tid))
        if t is None:
            continue
        if dept_set and (t.department is None or t.department not in dept_set):
            continue
        if prod_set and (t.product is None or t.product not in prod_set):
            continue
        results.append(
            {
                "ticket_id": int(t.id),
                "score": float(score_map.get(int(t.id), 0.0)),
                "department": t.department,
                "product": t.product,
                "summary": t.summary,
            }
        )
        if len(results) >= k:
            break

    payload = {
        "dataset_id": dataset_id,
        "k": int(k),
        "backend": backend_choice or (None),
        "model_name": model_name,
        "rerank": bool(rerank_flag),
        "rerank_backend": rerank_backend_norm,
        "results": results,
    }
    return JSONResponse(payload)