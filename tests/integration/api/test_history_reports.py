from __future__ import annotations

import io
from typing import Any, Dict, List

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from db.models import Base
from db.session import engine


def _make_csv_bytes() -> bytes:
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
                "department": "Finance",
                "Assignment_Group": "Accounts",
                "Product": "ERP",
                "Summary": "Request access to ERP module",
                "Quality": "poor",
                "Resolution Complexity": "high",
                "Reassignment_group_count_tracking_index": 3,
            },
        ]
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def test_history_and_reports_endpoints_work_with_offline_analyzer():
    # Fresh DB
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    app = create_app()
    client = TestClient(app)

    # Upload dataset
    csv_bytes = _make_csv_bytes()
    files = {"file": ("tickets.csv", csv_bytes, "text/csv")}
    resp_ingest = client.post("/ingest/upload", files=files)
    assert resp_ingest.status_code == 200, resp_ingest.text
    payload_ingest = resp_ingest.json()

    dataset_id = int(payload_ingest["dataset_id"])
    assert int(payload_ingest["row_count"]) == 3

    # Run two analyses (offline backend)
    req1 = {
        "dataset_id": dataset_id,
        "question": "Identify onboarding opportunities",
        "prompt_version": "v1",
        "analyzer_backend": "offline",
        "max_tickets": 3,
        "token_budget": 256,
    }
    resp_analysis1 = client.post("/analysis/run", json=req1)
    assert resp_analysis1.status_code == 200, resp_analysis1.text
    payload_analysis1 = resp_analysis1.json()
    aid1 = int(payload_analysis1["analysis_id"])

    req2 = {
        "dataset_id": dataset_id,
        "question": "Focus areas for training",
        "prompt_version": "v2",
        "analyzer_backend": "offline",
        "max_tickets": 2,
        "token_budget": 128,
    }
    resp_analysis2 = client.post("/analysis/run", json=req2)
    assert resp_analysis2.status_code == 200, resp_analysis2.text
    payload_analysis2 = resp_analysis2.json()
    aid2 = int(payload_analysis2["analysis_id"])

    # History: pagination
    resp_hist_page1 = client.get(
        "/history/analyses",
        params={"dataset_id": dataset_id, "limit": 1, "offset": 0},
    )
    assert resp_hist_page1.status_code == 200, resp_hist_page1.text
    page1: Dict[str, Any] = resp_hist_page1.json()
    assert int(page1["limit"]) == 1
    assert int(page1["offset"]) == 0
    assert int(page1["total"]) >= 2
    items1: List[Dict[str, Any]] = page1.get("items", [])
    assert len(items1) == 1

    # Next page
    resp_hist_page2 = client.get(
        "/history/analyses",
        params={"dataset_id": dataset_id, "limit": 1, "offset": 1},
    )
    assert resp_hist_page2.status_code == 200, resp_hist_page2.text
    page2: Dict[str, Any] = resp_hist_page2.json()
    items2: List[Dict[str, Any]] = page2.get("items", [])
    assert len(items2) == 1
    # IDs across pages should be distinct (order: latest first)
    assert items1[0]["id"] != items2[0]["id"]

    # Filter by prompt_version=v1 should include at least the first analysis
    resp_hist_v1 = client.get(
        "/history/analyses",
        params={"dataset_id": dataset_id, "prompt_version": "v1", "limit": 10, "offset": 0},
    )
    assert resp_hist_v1.status_code == 200, resp_hist_v1.text
    page_v1: Dict[str, Any] = resp_hist_v1.json()
    items_v1: List[Dict[str, Any]] = page_v1.get("items", [])
    assert any(int(it["id"]) == aid1 for it in items_v1)

    # Reports: assemble markdown and counts
    resp_report = client.get(f"/reports/{dataset_id}")
    assert resp_report.status_code == 200, resp_report.text
    payload_report: Dict[str, Any] = resp_report.json()
    assert int(payload_report["dataset_id"]) == dataset_id
    assert isinstance(payload_report.get("analysis_count"), int) and payload_report["analysis_count"] >= 2
    md = (payload_report.get("report_markdown") or "").strip()
    assert md, "report_markdown should be non-empty"
    assert "## Recent Analyses" in md
    assert "## Current Analytics Snapshot" in md