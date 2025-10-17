from __future__ import annotations

from typing import TypedDict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from db.models import Dataset
from db.repositories.clusters_repo import ClustersRepository
from db.repositories.embeddings_repo import EmbeddingsRepository
from db.session import get_session
from engine.analytics.clustering import hdbscan_cluster, kmeans_cluster, tfidf_top_terms
from vector_store.faiss_index import FaissIndexAdapter

router = APIRouter(prefix="/cluster", tags=["clustering"])


class ClusterRunResponse(TypedDict, total=False):
    dataset_id: int
    algorithm: str
    model_name: str
    run_id: int
    silhouette: float | None
    cluster_counts: dict[int, int]


def _infer_model_name_from_faiss(dataset_id: int) -> str | None:
    """
    Best-effort inference of model_name used for embeddings via FAISS metadata.
    Returns None if unavailable.
    """
    index = FaissIndexAdapter()
    meta = index._read_meta(dataset_id)  # type: ignore[attr-defined]
    if not isinstance(meta, dict):
        return None
    m = str(meta.get("model_name") or "").strip()
    return m or None


@router.post("/run", response_class=JSONResponse)
async def run_clustering(request: Request, db: Session = Depends(get_session)) -> JSONResponse:
    """
    Cluster persisted embeddings for a dataset using K-Means or HDBSCAN and store results.

    Request body (JSON) is parsed manually to avoid Pydantic TypeAdapter issues:
      {
        "dataset_id": int,
        "algorithm": "kmeans" | "hdbscan",
        "params": {
          "n_clusters"?: int,
          "min_cluster_size"?: int,
          "min_samples"?: int
        },
        "model_name"?: str
      }

    Flow:
      - Validate dataset exists.
      - Resolve model_name (prefer request; fallback to FAISS metadata).
      - Verify embeddings exist for dataset+model_name.
      - Fetch (ids, vectors) and candidate texts aligned to ids.
      - Run selected clustering via pure Engine functions.
      - Persist run, assignments, metrics (silhouette), and TF-IDF top terms (skip noise).
      - Return summary: dataset_id, algorithm, model_name, run_id, silhouette, cluster_counts.

    Error mapping:
      - 404 when dataset or embeddings missing.
      - 400 invalid params (e.g., n_clusters ≤ 1).
      - 422 insufficient vectors or unavailable algorithm dependency.
      - 500 unexpected errors with sanitized message.
    """
    # Parse JSON body defensively
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from e

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    # dataset_id
    dataset_id_val = body.get("dataset_id", None)
    if dataset_id_val is None:
        raise HTTPException(status_code=400, detail="dataset_id is required")
    try:
        dataset_id_int = int(dataset_id_val)
    except Exception as e:
        raise HTTPException(status_code=400, detail="dataset_id must be an integer") from e

    # Validate dataset existence
    dataset = db.get(Dataset, dataset_id_int)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id_int} not found")

    # Algorithm
    algorithm_in = str(body.get("algorithm", "") or "").strip().lower()
    if algorithm_in not in ("kmeans", "hdbscan"):
        raise HTTPException(status_code=400, detail="algorithm must be 'kmeans' or 'hdbscan'")

    # Params
    params_in = body.get("params") or {}
    if not isinstance(params_in, dict):
        raise HTTPException(status_code=400, detail="params must be an object")

    # Resolve model_name (prefer request; else FAISS metadata)
    model_name_in = body.get("model_name")
    model_name = None
    if isinstance(model_name_in, str) and model_name_in.strip():
        model_name = model_name_in.strip()
    else:
        model_name = _infer_model_name_from_faiss(dataset.id)
    if not model_name:
        # Without a model_name we cannot fetch embeddings
        raise HTTPException(
            status_code=404, detail="Embeddings model_name could not be resolved for dataset"
        )

    # Verify embeddings exist
    if not EmbeddingsRepository.exists_for_dataset(
        db, dataset_id=dataset.id, model_name=model_name
    ):
        raise HTTPException(
            status_code=404,
            detail=f"Embeddings for dataset {dataset.id} and model '{model_name}' not found",
        )

    # Fetch embeddings and align texts to ids for TF-IDF
    ids_all, vecs_all = EmbeddingsRepository.fetch_by_dataset(
        db, dataset_id=dataset.id, model_name=model_name
    )
    if not ids_all or not vecs_all or len(ids_all) != len(vecs_all):
        raise HTTPException(
            status_code=422, detail="No embeddings vectors available for clustering"
        )

    # Build text map for aligned TF-IDF input
    ticket_ids_texts, candidate_texts = EmbeddingsRepository.fetch_candidate_texts(
        db, dataset_id=dataset.id, limit=250_000
    )
    text_map: dict[int, str] = {}
    for tid, text in zip(ticket_ids_texts, candidate_texts, strict=False):
        text_map[int(tid)] = str(text or "").strip()
    texts_aligned: list[str] = [text_map.get(int(tid), "") for tid in ids_all]

    # Run clustering
    try:
        if algorithm_in == "kmeans":
            n_clusters_val = params_in.get("n_clusters", None)
            if n_clusters_val is None:
                raise HTTPException(
                    status_code=400, detail="params.n_clusters is required for kmeans"
                )
            try:
                n_clusters_int = int(n_clusters_val)
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail="params.n_clusters must be an integer"
                ) from e

            result = kmeans_cluster(vectors=vecs_all, n_clusters=n_clusters_int, random_state=42)
        else:
            # hdbscan path
            min_cluster_size = params_in.get("min_cluster_size", 5)
            min_samples = params_in.get("min_samples", 5)
            try:
                min_cluster_size_int = int(min_cluster_size)
                min_samples_int = int(min_samples)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail="params.min_cluster_size and min_samples must be integers",
                ) from e

            try:
                result = hdbscan_cluster(
                    vectors=vecs_all,
                    min_cluster_size=min_cluster_size_int,
                    min_samples=min_samples_int,
                )
            except RuntimeError as e:
                # Optional dependency unavailable
                raise HTTPException(status_code=422, detail=f"HDBSCAN unavailable: {e}") from e
    except ValueError as e:
        # Input/parameter validation errors from engine
        msg = str(e)
        if "insufficient vectors" in msg.lower():
            raise HTTPException(status_code=422, detail=msg) from e
        raise HTTPException(status_code=400, detail=msg) from e
    except HTTPException:
        # Reraise mapped HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {e}") from e

    assignments: list[int] = list(result.get("assignments") or [])
    silhouette: float | None = (
        result.get("silhouette") if isinstance(result.get("silhouette"), (float, int)) else None
    )

    if len(assignments) != len(ids_all):
        raise HTTPException(status_code=500, detail="Engine returned invalid assignments length")

    # Persist run
    try:
        run_id = ClustersRepository.create_run(
            db,
            dataset_id=dataset.id,
            model_name=model_name,
            algorithm=algorithm_in,
            params={
                "algorithm": algorithm_in,
                **(
                    {"n_clusters": int(params_in.get("n_clusters"))}
                    if algorithm_in == "kmeans" and params_in.get("n_clusters") is not None
                    else {}
                ),
                **(
                    {"min_cluster_size": int(params_in.get("min_cluster_size", 5))}
                    if algorithm_in == "hdbscan"
                    else {}
                ),
                **(
                    {"min_samples": int(params_in.get("min_samples", 5))}
                    if algorithm_in == "hdbscan"
                    else {}
                ),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create cluster run: {e}") from e

    # Persist assignments
    try:
        tuples: list[tuple[int, int]] = [
            (int(tid), int(cid)) for tid, cid in zip(ids_all, assignments, strict=False)
        ]
        ClustersRepository.store_assignments(db, run_id=run_id, assignments=tuples, batch_size=1000)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store assignments: {e}") from e

    # Persist metrics (silhouette)
    try:
        ClustersRepository.store_metrics(db, run_id=run_id, metrics={"silhouette": silhouette})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store metrics: {e}") from e

    # Compute and persist TF-IDF top terms per cluster (skip noise)
    try:
        top_terms = tfidf_top_terms(texts=texts_aligned, assignments=assignments, top_k=10)
        ClustersRepository.store_top_terms(db, run_id=run_id, top_terms=top_terms, batch_size=1000)
    except Exception as e:
        # Do not fail the entire run if TF-IDF fails; but surface error for observability
        raise HTTPException(
            status_code=500, detail=f"Failed to compute/store top terms: {e}"
        ) from e

    # Build response summary
    try:
        summary = ClustersRepository.fetch_run_summary(db, run_id=run_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch run summary: {e}") from e

    payload: ClusterRunResponse = {
        "dataset_id": dataset.id,
        "algorithm": algorithm_in,
        "model_name": model_name,
        "run_id": int(run_id),
        "silhouette": summary.get("silhouette"),
        "cluster_counts": summary.get("cluster_counts", {}),
    }
    return JSONResponse(payload)
