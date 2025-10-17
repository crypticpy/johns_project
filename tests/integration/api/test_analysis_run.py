from __future__ import annotations

import io

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from db.models import Base
from db.session import engine, SessionLocal
from db.repositories.analyses_repo import AnalysesRepository


def _make_csv_bytes() -> bytes:
    # Construct a small DataFrame with canonical and variant columns
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
            # Include variant columns to test normalization; repository maps these at ingest time
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


def test_analysis_run_persists_record_and_offline_markdown_structure():
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
    row_count = int(payload_ingest["row_count"])
    assert row_count == 3

    # Run analysis with offline backend and bounded sampling/token budget
    req_body = {
        "dataset_id": dataset_id,
        "question": "Identify onboarding opportunities for top departments.",
        "prompt_version": "v1",
        "analyzer_backend": "offline",
        "max_tickets": 4,
        "token_budget": 256,
    }
    resp_analysis = client.post("/analysis/run", json=req_body)
    assert resp_analysis.status_code == 200, resp_analysis.text
    payload_analysis = resp_analysis.json()

    assert "analysis_id" in payload_analysis and isinstance(payload_analysis["analysis_id"], int)
    assert int(payload_analysis["dataset_id"]) == dataset_id
    assert payload_analysis["prompt_version"] == "v1"
    assert isinstance(payload_analysis.get("ticket_count"), int)

    analysis_id = int(payload_analysis["analysis_id"])

    # Verify persisted record via repository and check structured markdown headings
    db = SessionLocal()
    try:
        rows = AnalysesRepository.list_analyses(db, limit=10, offset=0, dataset_id=dataset_id)
        assert any(int(r.id) == analysis_id for r in rows)

        # Fetch the specific record to inspect content
        rec = next(r for r in rows if int(r.id) == analysis_id)
        md = (rec.result_markdown or "").strip()
        assert md, "result_markdown should be non-empty"
        # Expected headings from offline analyzer
        assert "# Service Desk Analysis" in md
        assert "Prompt Version: v1" in md
        assert "## Summary" in md
        assert "## Distributions" in md
        assert "## Insights" in md
        assert "## Recommendations" in md
        assert "## Onboarding Key-Values" in md
    finally:
        db.close()