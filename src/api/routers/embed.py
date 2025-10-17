from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ai.embeddings.factory import resolve_defaults, select_embedder
from db.models import Dataset
from db.repositories.embeddings_repo import EmbeddingsRepository
from db.session import get_session
from vector_store.faiss_index import FaissIndexAdapter, FaissIndexError


router = APIRouter(prefix="/embed", tags=["embeddings"])


@router.post("/run", response_class=JSONResponse)
async def run_embeddings(request: Request, db: Session = Depends(get_session)) -> JSONResponse:
    """
    Compute and persist embeddings for a dataset, then (re)build the FAISS index.

    Request body (JSON) is parsed manually to avoid Pydantic TypeAdapter issues:
      {
        "dataset_id": int,
        "model_name": str | null,
        "batch_size": int | null,
        "backend": "sentence-transformers" | "builtin" | null
      }

    - Validates dataset exists.
    - Uses normalized_text if present, else summary for embedding input.
    - Idempotent persistence per (dataset_id, model_name, ticket_id).
    - Index is rebuilt from DB embeddings to avoid duplicates.

    Response:
      {dataset_id, model_name, backend, vector_dim, embedded_count, indexed: true}
    """
    # Parse JSON body defensively
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    dataset_id_val = body.get("dataset_id", None)
    if dataset_id_val is None:
        raise HTTPException(status_code=400, detail="dataset_id is required")
    try:
        dataset_id_int = int(dataset_id_val)
    except Exception:
        raise HTTPException(status_code=400, detail="dataset_id must be an integer")

    model_name_in = body.get("model_name", None)
    batch_size_in = body.get("batch_size", None)
    backend_in = body.get("backend", None)

    # Validate dataset existence
    dataset = db.get(Dataset, dataset_id_int)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id_int} not found")

    # Resolve backend, model, batch size
    embedder = select_embedder(backend_in)
    default_model, default_batch = resolve_defaults()
    model_name = (model_name_in or default_model).strip()
    try:
        batch_size = int(batch_size_in or default_batch)
    except Exception:
        raise HTTPException(status_code=400, detail="batch_size must be an integer when provided")

    # Fetch candidate texts (ids and input strings)
    ticket_ids, texts = EmbeddingsRepository.fetch_candidate_texts(db, dataset_id=dataset.id)
    if not ticket_ids or not texts or len(ticket_ids) != len(texts):
        # Graceful no-op response
        return JSONResponse(
            {
                "dataset_id": dataset.id,
                "model_name": model_name,
                "backend": (backend_in or "sentence-transformers"),
                "vector_dim": 0,
                "embedded_count": 0,
                "indexed": False,
            }
        )

    # Compute embeddings (adapter handles batching/backpressure)
    try:
        vectors = embedder.embed_texts(texts, model=model_name, batch_size=batch_size)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding backend error: {e}")

    if not vectors or len(vectors) != len(ticket_ids):
        raise HTTPException(status_code=500, detail="Embedding adapter returned invalid vector count")

    # Persist embeddings idempotently
    try:
        counts = EmbeddingsRepository.upsert_for_dataset(
            db,
            dataset_id=dataset.id,
            model_name=model_name,
            records=list(zip(ticket_ids, vectors)),
            batch_size=1000,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist embeddings: {e}")

    # Rebuild FAISS index from persisted embeddings to avoid duplicates
    ids_all, vecs_all = EmbeddingsRepository.fetch_by_dataset(db, dataset_id=dataset.id, model_name=model_name)
    vector_dim: int = 0 if not vecs_all else len(vecs_all[0])

    index = FaissIndexAdapter()
    try:
        if ids_all and vecs_all:
            index.build_index(dataset_id=dataset.id, vectors=vecs_all, ids=ids_all, model_name=model_name)
    except FaissIndexError as e:
        raise HTTPException(status_code=500, detail=f"FAISS index error: {e}")

    payload = {
        "dataset_id": dataset.id,
        "model_name": model_name,
        "backend": (backend_in or "sentence-transformers"),
        "vector_dim": vector_dim,
        "embedded_count": len(ticket_ids),  # total processed rows (idempotent across runs)
        "indexed": True,
    }
    return JSONResponse(payload)