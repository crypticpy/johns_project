from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from db.models import Dataset, Ticket
from db.repositories.tickets_repo import TicketsRepository
from db.session import get_session
from engine.analytics.metrics import (
    compute_complexity_distribution,
    compute_department_volume,
    compute_product_distribution,
    compute_quality_distribution,
    compute_reassignment_distribution,
)
from engine.analytics.visualizations import transform_metrics

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _tickets_to_df(tickets: list[Ticket]) -> pd.DataFrame:
    """
    Build a DataFrame with canonical columns from Ticket ORM rows.
    Canonical columns:
      - Department
      - extract_product
      - ticket_quality
      - resolution_complexity
      - Reassignment group count tracking_index
      - summarize_ticket (included for completeness; not used by metrics)
    """
    if not tickets:
        # Construct an empty DataFrame with canonical columns to satisfy downstream expectations
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
    for t in tickets:
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


@router.get("/metrics", response_class=JSONResponse)
async def get_metrics(request: Request, db: Session = Depends(get_session)) -> JSONResponse:
    """
    Return analytics metrics for a dataset.

    Query params:
      - dataset_id: int (required)
      - top_n: int (default=10) for department_volume ranking
      - department_filter: optional repeated param to filter tickets by Department

    Flow:
      - Validate dataset exists
      - Retrieve dataset tickets; apply optional department_filter
      - Build a pandas DataFrame with canonical columns
      - Compute metrics via engine.analytics.metrics
      - Structure payload via engine.analytics.visualizations.transform_metrics

    Response:
      {
        "dataset_id": int,
        "metrics": {
          "quality": {type, title, labels, values},
          "complexity": {type, title, labels, values},
          "department_volume": {type, title, labels, values},
          "reassignment": {type, title, buckets, counts},
          "product": {type, title, labels, values}
        }
      }
    """
    # Parse query params defensively to avoid Pydantic TypeAdapter issues
    qp = request.query_params
    dataset_id_val = qp.get("dataset_id")
    if dataset_id_val is None:
        raise HTTPException(status_code=400, detail="dataset_id is required")
    try:
        dataset_id = int(dataset_id_val)
    except Exception as e:
        raise HTTPException(status_code=400, detail="dataset_id must be an integer") from e

    top_n_val = qp.get("top_n", "10")
    try:
        top_n = int(top_n_val)
    except Exception as e:
        raise HTTPException(status_code=400, detail="top_n must be an integer") from e

    # Repeated param support
    departments = list(qp.getlist("department_filter")) or None

    # Validate dataset existence
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    # Fetch tickets with optional department filter
    tickets: list[Ticket] = TicketsRepository.query_filtered(
        db,
        dataset_id=dataset_id,
        departments=departments,
        limit=100_000,
        offset=0,
    )

    # Build DataFrame and compute metrics (pure functions; fallback-safe)
    df = _tickets_to_df(tickets)

    raw_metrics: dict[str, Any] = {
        "quality": compute_quality_distribution(df),
        "complexity": compute_complexity_distribution(df),
        "department_volume": compute_department_volume(df, top_n=top_n),
        "reassignment": compute_reassignment_distribution(df),
        "product": compute_product_distribution(df),
    }

    # Transform into chart-agnostic specs
    metrics_spec = transform_metrics(raw_metrics)

    payload = {
        "dataset_id": int(dataset_id),
        "metrics": metrics_spec,
    }
    return JSONResponse(payload)
