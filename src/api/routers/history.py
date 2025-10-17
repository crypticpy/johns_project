from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from db.repositories.analyses_repo import AnalysesRepository
from db.session import get_session
from security.auth import require_roles

router = APIRouter(prefix="/history", tags=["history"])


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    s = str(value).strip()
    try:
        # Accept both full ISO and date-only
        if len(s) <= 10 and "-" in s and "T" not in s:
            return datetime.fromisoformat(s + "T00:00:00")
        return datetime.fromisoformat(s)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ISO datetime: {value}") from e


@router.get("/analyses", response_class=JSONResponse)
async def list_analyses_history(
    request: Request,
    db: Session = Depends(get_session),
    claims: dict[str, Any] = Depends(require_roles({"viewer", "admin"})),
) -> JSONResponse:
    """
    Paginated analyses history with filters.

    Implementation note:
    - Parse query params from Request.query_params to avoid Pydantic TypeAdapter edge-cases.
    """
    qp = request.query_params

    # Pagination
    limit_val = qp.get("limit", "50")
    offset_val = qp.get("offset", "0")
    try:
        limit = int(limit_val)
        offset = int(offset_val)
    except Exception as e:
        raise HTTPException(status_code=400, detail="limit and offset must be integers") from e
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    # Optional filters
    dataset_id: int | None = None
    dataset_id_val = qp.get("dataset_id")
    if dataset_id_val is not None:
        try:
            dataset_id = int(dataset_id_val)
        except Exception as e:
            raise HTTPException(status_code=400, detail="dataset_id must be an integer") from e

    prompt_version = qp.get("prompt_version") or None

    date_from_raw = qp.get("date_from") or None
    date_to_raw = qp.get("date_to") or None
    df = _parse_iso_datetime(date_from_raw)
    dt = _parse_iso_datetime(date_to_raw)

    rows = AnalysesRepository.list_analyses(
        db,
        limit=limit,
        offset=offset,
        dataset_id=dataset_id,
        prompt_version=prompt_version,
        date_from=df,
        date_to=dt,
    )
    total = AnalysesRepository.count_analyses(
        db,
        dataset_id=dataset_id,
        prompt_version=prompt_version,
        date_from=df,
        date_to=dt,
    )

    items: list[dict[str, Any]] = []
    for r in rows:
        items.append(
            {
                "id": int(r.id),
                "dataset_id": int(r.dataset_id),
                "prompt_version": r.prompt_version,
                "question": r.question,
                "ticket_count": int(r.ticket_count or 0),
                "created_at": r.created_at.isoformat() if getattr(r, "created_at", None) else None,
            }
        )

    return JSONResponse(
        {
            "limit": int(limit),
            "offset": int(offset),
            "total": int(total),
            "items": items,
        }
    )
