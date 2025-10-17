from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ai.llm.factory import select_analyzer
from ai.llm.interface import AnalyzerError
from config.settings import get_settings
from db.models import Analysis, Dataset, Ticket
from db.repositories.analyses_repo import AnalysesRepository
from db.repositories.audit_repo import AuditRepository
from db.repositories.tickets_repo import TicketsRepository
from db.session import get_session
from engine.analytics.metrics import (
    compute_complexity_distribution,
    compute_department_volume,
    compute_quality_distribution,
)
from engine.features.sampling import SamplingConfig, stratified_sample
from security.auth import require_roles

router = APIRouter(prefix="/analysis", tags=["analysis"])


def _tickets_to_df(tickets: list[Ticket]) -> pd.DataFrame:
    """
    Build a DataFrame w/ canonical columns expected by sampling and metrics.

    Canonical columns:
      - id (ticket id for stable IDs in sample output)
      - Department
      - extract_product
      - ticket_quality
      - resolution_complexity
      - Reassignment group count tracking_index
      - summarize_ticket
    """
    if not tickets:
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
    for t in tickets:
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


def _estimate_tokens_from_text(text: str) -> int:
    # Heuristic: tokens ~= chars/4
    try:
        return max(0, int(len(text) / 4))
    except Exception:
        return 0


def _build_comparison_section(db: Session, compare_dataset_id: int) -> tuple[str, dict[str, Any]]:
    """
    Build a compact comparison summary section for the second dataset.
    Returns (markdown, metrics_summary_dict).
    """
    ds2 = db.get(Dataset, int(compare_dataset_id))
    if ds2 is None:
        raise HTTPException(
            status_code=404, detail=f"Comparison dataset {compare_dataset_id} not found"
        )

    tickets2: list[Ticket] = TicketsRepository.query_filtered(
        db, dataset_id=int(compare_dataset_id), limit=100_000, offset=0
    )
    df2 = _tickets_to_df(tickets2)

    top_depts2 = compute_department_volume(df2, top_n=5)
    quality2 = compute_quality_distribution(df2)
    complexity2 = compute_complexity_distribution(df2)

    lines: list[str] = []
    lines.append(f"## Comparison Dataset Summary (dataset_id={compare_dataset_id})")
    total2 = int(len(df2))
    lines.append(f"Total tickets: {total2}")
    if top_depts2:
        lines.append("Top Departments: " + ", ".join([f"{d}({c})" for d, c in top_depts2]))
    if quality2:
        lines.append("Quality: " + ", ".join([f"{k}({v})" for k, v in sorted(quality2.items())]))
    if complexity2:
        lines.append(
            "Complexity: " + ", ".join([f"{k}({v})" for k, v in sorted(complexity2.items())])
        )

    metrics_summary = {
        "total_rows": total2,
        "top_departments": top_depts2,
        "quality": quality2,
        "complexity": complexity2,
    }
    return ("\n".join(lines).strip(), metrics_summary)


def _derive_departments_from_segments(segments: list[dict[str, Any]]) -> list[str]:
    depts: set[str] = set()
    for seg in segments or []:
        keymap = seg.get("key") or {}
        dep = keymap.get("Department")
        if isinstance(dep, str) and dep and dep != "(missing)":
            depts.add(dep)
    return sorted(depts)


@router.post("/run", response_class=JSONResponse)
async def run_analysis(
    request: Request,
    db: Session = Depends(get_session),
) -> JSONResponse:
    """
    Execute an analysis run:
      - Load dataset tickets
      - Perform stratified sampling with token budget
      - Build prompt context (optionally include comparison dataset summary)
      - Select analyzer backend and run analysis
      - Persist analysis record and return metadata

    Request JSON:
      {
        "dataset_id": int,
        "question": str,
        "prompt_version": str,               # optional; defaults to "v1"
        "analyzer_backend": "openai"|"offline" | null,
        "max_tickets": int | null,           # optional; default 50
        "token_budget": int | null,          # optional; default 2000
        "compare_dataset_id": int | null     # optional
      }

    Response:
      { "analysis_id", "dataset_id", "prompt_version", "ticket_count", "created_at" }
    """
    # Conditional RBAC guard
    settings = get_settings()
    claims: dict[str, Any] = {}
    if getattr(settings, "enable_rbac", False):
        checker = require_roles({"analyst", "admin"})
        claims = await checker(request)

    # Parse JSON body defensively
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from e

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    dataset_id_val = body.get("dataset_id")
    if dataset_id_val is None:
        raise HTTPException(status_code=400, detail="dataset_id is required")
    try:
        dataset_id = int(dataset_id_val)
    except Exception as e:
        raise HTTPException(status_code=400, detail="dataset_id must be an integer") from e

    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    prompt_version = (body.get("prompt_version") or "v1").strip()
    if not prompt_version:
        prompt_version = "v1"

    analyzer_backend = body.get("analyzer_backend", None)
    if analyzer_backend is not None:
        if str(analyzer_backend).strip().lower() not in ("openai", "offline"):
            raise HTTPException(
                status_code=400, detail="analyzer_backend must be 'openai' or 'offline'"
            )

    max_tickets = body.get("max_tickets", 50)
    token_budget = body.get("token_budget", 2000)
    compare_dataset_id = body.get("compare_dataset_id", None)
    comparison_mode = compare_dataset_id is not None

    # Validate dataset(s)
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    if comparison_mode:
        # Validate existence; actual summary built later
        ds2 = db.get(Dataset, int(compare_dataset_id))
        if ds2 is None:
            raise HTTPException(
                status_code=404, detail=f"Comparison dataset {compare_dataset_id} not found"
            )

    # Load tickets and build DataFrame
    tickets: list[Ticket] = TicketsRepository.query_filtered(
        db, dataset_id=dataset_id, limit=100_000, offset=0
    )
    df = _tickets_to_df(tickets)

    # Perform stratified sampling
    try:
        cfg = SamplingConfig(
            max_tickets=int(max_tickets or 0),
            token_budget=int(token_budget or 0),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid sampling configuration") from e

    try:
        sample = stratified_sample(df, cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sampling failed: {e}") from e

    # Build prompt context
    context_text = str(sample.get("context_text") or "").strip()
    if not context_text:
        context_text = "# Analysis Context\n(no data)"

    comparison_metrics: dict[str, Any] = {}
    if comparison_mode:
        try:
            compare_md, compare_summary = _build_comparison_section(db, int(compare_dataset_id))
            # Append with a blank line separator
            context_text = f"{context_text}\n\n{compare_md}"
            comparison_metrics = {"comparison": compare_summary}
        except HTTPException:
            raise
        except Exception as e:
            # Do not abort analysis if comparison summary fails; proceed without it
            comparison_metrics = {"comparison_error": str(e)}

    # Select analyzer and run
    try:
        analyzer = select_analyzer(
            str(analyzer_backend).strip().lower() if analyzer_backend else None
        )  # type: ignore[arg-type]
    except AnalyzerError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        result_md = analyzer.analyze(
            context=context_text,
            question=question,
            prompt_version=prompt_version,
            comparison_mode=bool(comparison_mode),
        )
    except AnalyzerError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyzer error: {e}") from e

    # Persist analysis
    sampled_ids: list[int] = list(sample.get("sampled_ids") or [])
    segments: list[dict[str, Any]] = list(sample.get("segments") or [])
    summary: dict[str, Any] = dict(sample.get("summary") or {})
    departments_used = _derive_departments_from_segments(segments)

    metrics: dict[str, Any] = {
        "sampling_summary": summary,
        "segments": segments,
        "estimated_prompt_tokens": _estimate_tokens_from_text(context_text),
    }
    if comparison_metrics:
        metrics.update(comparison_metrics)

    filters: dict[str, Any] = {}
    if departments_used:
        filters["departments"] = departments_used

    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist analysis: {e}") from e

    # Emit audit log (non-blocking; avoid PII)
    try:
        subject = str(
            claims.get("sub") or claims.get("subject") or claims.get("user") or "anonymous"
        )
        AuditRepository.record(
            db,
            subject=subject,
            action="analysis.run",
            resource=f"dataset:{dataset_id}",
            metadata={
                "analysis_id": int(analysis_id),
                "prompt_version": str(prompt_version),
            },
        )
    except Exception:
        # Do not fail request due to audit logging
        pass

    # Fetch created_at for response summary
    created_at_iso: str | None = None
    try:
        row = db.get(Analysis, int(analysis_id))
        if row and isinstance(row.created_at, datetime):
            created_at_iso = row.created_at.isoformat()
    except Exception:
        created_at_iso = None

    payload = {
        "analysis_id": int(analysis_id),
        "dataset_id": int(dataset_id),
        "prompt_version": str(prompt_version),
        "ticket_count": int(len(sampled_ids)),
        "created_at": created_at_iso,
    }
    return JSONResponse(payload)
