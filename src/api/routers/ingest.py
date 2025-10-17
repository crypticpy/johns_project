from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from db.repositories.datasets_repo import DatasetsRepository
from db.repositories.tickets_repo import TicketsRepository
from db.session import get_session
from engine.ingest.loader import validate_and_load

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/upload", response_class=JSONResponse)
async def upload_dataset(request: Request, db: Session = Depends(get_session)) -> JSONResponse:
    """
    Accept a CSV/Excel upload, normalize columns, persist dataset and tickets, and return summary.

    Summary JSON:
      { dataset_id, name, row_count, department_count, file_hash }
    """
    # Parse multipart form defensively to avoid Pydantic TypeAdapter issues with UploadFile
    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to parse multipart form") from e

    # Try canonical field name first
    file_obj = form.get("file")
    # Fallback: locate first file-like entry (robust to varying field names)
    if not (hasattr(file_obj, "filename") and hasattr(file_obj, "read")):
        file_obj = None
        for v in form.values():
            if hasattr(v, "filename") and hasattr(v, "read"):
                file_obj = v
                break

    if file_obj is None:
        raise HTTPException(status_code=400, detail="File field is required")

    # Validate filename
    filename = (getattr(file_obj, "filename", "") or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Read all bytes
    try:
        file_bytes = await file_obj.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file") from e

    # Load and normalize
    try:
        df, meta = validate_and_load(file_bytes, filename)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not isinstance(df, pd.DataFrame):
        raise HTTPException(
            status_code=400, detail="Uploaded content did not parse into a DataFrame"
        )

    file_hash: str = str(meta["file_hash"])
    row_count: int = int(meta["rows"])

    # Create or get dataset by hash (idempotent)
    dataset = DatasetsRepository.create_or_get(
        db,
        name=filename,
        file_hash=file_hash,
        row_count=row_count,
        department_count=0,
        metadata={"filename": filename},
    )

    inserted = 0
    # If this is a new dataset (no tickets yet), insert tickets
    # Guard: If dataset already existed, avoid duplicating tickets on re-upload
    # We check whether tickets exist for this dataset before bulk insert.
    existing_tickets = TicketsRepository.query_filtered(db, dataset_id=dataset.id, limit=1)
    if not existing_tickets and row_count > 0:
        inserted = TicketsRepository.bulk_insert(db, dataset_id=dataset.id, df=df)

    # Recompute and persist distinct department count
    dept_count = DatasetsRepository.recompute_department_count(db, dataset.id)

    payload: dict[str, Any] = {
        "dataset_id": dataset.id,
        "name": dataset.name,
        "row_count": dataset.row_count,
        "department_count": dept_count,
        "file_hash": dataset.file_hash,
        "inserted_tickets": inserted,
    }
    return JSONResponse(payload)
