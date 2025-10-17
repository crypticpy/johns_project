from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from db.models import Dataset, Ticket
from db.repositories.analyses_repo import AnalysesRepository
from db.repositories.tickets_repo import TicketsRepository
from db.session import get_session
from engine.analytics.metrics import (
    compute_complexity_distribution,
    compute_department_volume,
    compute_product_distribution,
    compute_quality_distribution,
    compute_reassignment_distribution,
)

router = APIRouter(prefix="/reports", tags=["reports"])


def _tickets_to_df(tickets: List[Ticket]) -> pd.DataFrame:
    """
    Build a DataFrame with canonical columns for metrics snapshot.
    """
    if not tickets:
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
    records: List[Dict[str, Any]] = []
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


def _metrics_to_markdown(df: pd.DataFrame) -> str:
    """
    Convert current metrics snapshot into concise Markdown sections.
    """
    lines: List[str] = []
    lines.append("## Current Analytics Snapshot")
    lines.append(f"Total Tickets: {int(len(df))}")

    dept_vol: List[Tuple[str, int]] = compute_department_volume(df, top_n=10)
    if dept_vol:
        lines.append("### Top Departments by Volume")
        for dep, cnt in dept_vol:
            lines.append(f"- {dep}: {cnt}")
    else:
        lines.append("### Top Departments by Volume")
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

    return "\n".join(lines).strip()


@router.get("/{dataset_id}", response_class=JSONResponse)
async def get_dataset_report(request: Request, db: Session = Depends(get_session)) -> JSONResponse:
    """
    Assemble a markdown report for the dataset:
      - Include recent analyses summary (titles and counts)
      - Include current analytics metrics snapshot (distributions)

    Implementation note:
      Parse path params from Request to avoid Pydantic TypeAdapter edge-cases.
    """
    # Parse dataset_id from path params defensively
    raw_id = request.path_params.get("dataset_id")
    try:
        dataset_id = int(raw_id)
    except Exception:
        raise HTTPException(status_code=400, detail="dataset_id must be an integer")

    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    # Recent analyses (limit 10)
    analyses = AnalysesRepository.list_analyses(
        db, limit=10, offset=0, dataset_id=dataset_id, prompt_version=None, date_from=None, date_to=None
    )
    analysis_count = len(analyses)

    # Build analyses summary section
    report_lines: List[str] = []
    report_lines.append(f"# Dataset Report (dataset_id={dataset_id})")
    report_lines.append("")
    report_lines.append("## Recent Analyses")
    if analyses:
        for a in analyses:
            title = (a.question or "").strip()
            pv = (a.prompt_version or "").strip()
            created = a.created_at.isoformat() if getattr(a, "created_at", None) else "n/a"
            report_lines.append(f"- [{created}] prompt={pv} tickets={int(a.ticket_count or 0)} — {title}")
    else:
        report_lines.append("- No analyses available.")

    # Current metrics snapshot
    tickets: List[Ticket] = TicketsRepository.query_filtered(db, dataset_id=dataset_id, limit=100_000, offset=0)
    df = _tickets_to_df(tickets)
    report_lines.append("")
    report_lines.append(_metrics_to_markdown(df))

    report_md = "\n".join(report_lines).strip()

    payload = {
        "dataset_id": int(dataset_id),
        "report_markdown": report_md,
        "analysis_count": int(analysis_count),
    }
    return JSONResponse(payload)