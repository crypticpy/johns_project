from __future__ import annotations

import io
from typing import Dict, List

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from db.models import Base
from db.session import engine


def _make_csv_bytes() -> bytes:
    """Construct a small canonical CSV in-memory for deterministic offline tests."""
    df = pd.DataFrame(
        [
            {
                "Department": "IT",
                "Assignment Group": "Service Desk",
                "extract_product": "Laptop",
                "summarize_ticket": "User cannot login",
                "ticket_quality": "good",
                "resolution_complexity": "low",
                "Reassignment group count tracking_index": 0,
            },
            {
                "Department": "HR",
                "Assignment Group": "Onboarding",
                "extract_product": "HRIS",
                "summarize_ticket": "Provision new employee account",
                "ticket_quality": "average",
                "resolution_complexity": "medium",
                "Reassignment group count tracking_index": 2,
            },
            {
                "Department": "Finance",
                "Assignment Group": "Accounts",
                "extract_product": "ERP",
                "summarize_ticket": "Request access to ERP module",
                "ticket_quality": "poor",
                "resolution_complexity": "high",
                "Reassignment group count tracking_index": 3,
            },
        ]
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def test_e2e_offline_pipeline_smoke() -> None:
    # Ensure clean database state for this test run (idempotent)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    app = create_app()
    client = TestClient(app)

    # 1) Ingest: upload CSV
    csv_bytes = _make_csv_bytes()
    files = {"file": ("tickets.csv", csv_bytes, "text/csv")}
    resp_ingest = client.post("/ingest/upload", files=files)
    assert resp_ingest.status_code == 200, resp_ingest.text
    payload_ingest = resp_ingest.json()

    dataset_id = int(payload_ingest["dataset_id"])
    assert dataset_id > 0
    assert int(payload_ingest["row_count"]) == 3

    # 2) Embeddings: run with builtin-384 (offline deterministic)
    req_embed = {
        "dataset_id": dataset_id,
        "backend": "builtin",
        "model_name": "builtin-384",
        "batch_size": 32,
    }
    resp_embed = client.post("/embed/run", json=req_embed)
    assert resp_embed.status_code == 200, resp_embed.text
    payload_embed = resp_embed.json()
    assert payload_embed["dataset_id"] == dataset_id
    assert int(payload_embed["vector_dim"]) == 384
    assert int(payload_embed["embedded_count"]) >= 1
    assert payload_embed["indexed"] is True

    # 3) Search: small k with a simple query string
    params = {"dataset_id": dataset_id, "q": "login", "k": "2"}
    resp_search = client.get("/search/nn", params=params)
    assert resp_search.status_code == 200, resp_search.text
    payload_search = resp_search.json()
    results: List[Dict] = list(payload_search.get("results") or [])
    assert len(results) >= 1

    # 4) Analysis: offline analyzer, prompt_version v1
    req_analysis = {
        "dataset_id": dataset_id,
        "question": "Identify top onboarding opportunities",
        "prompt_version": "v1",
        "analyzer_backend": "offline",
        "max_tickets": 50,
        "token_budget": 2000,
    }
    resp_analysis = client.post("/analysis/run", json=req_analysis)
    assert resp_analysis.status_code == 200, resp_analysis.text
    payload_analysis = resp_analysis.json()
    analysis_id = int(payload_analysis["analysis_id"])
    assert analysis_id > 0
    assert int(payload_analysis["dataset_id"]) == dataset_id
    assert (payload_analysis.get("prompt_version") or "") == "v1"
    assert int(payload_analysis.get("ticket_count", 0)) >= 1

    # 5) Report: retrieve markdown with analysis_count >= 1
    resp_report = client.get(f"/reports/{dataset_id}")
    assert resp_report.status_code == 200, resp_report.text
    payload_report = resp_report.json()
    report_md = (payload_report.get("report_markdown") or "").strip()
    assert isinstance(report_md, str) and len(report_md) > 0
    assert int(payload_report.get("analysis_count", 0)) >= 1