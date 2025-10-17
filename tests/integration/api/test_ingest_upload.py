from __future__ import annotations

import io

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from db.repositories.datasets_repo import DatasetsRepository
from db.repositories.tickets_repo import TicketsRepository
from db.session import SessionLocal, engine
from db.models import Base


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
            # Include some variant columns to test normalization behavior
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


def test_ingest_upload_persists_dataset_and_tickets_and_returns_summary():
    # Ensure a clean database for this test run (idempotent)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    app = create_app()
    client = TestClient(app)

    csv_bytes = _make_csv_bytes()
    files = {"file": ("tickets.csv", csv_bytes, "text/csv")}
    resp = client.post("/ingest/upload", files=files)

    assert resp.status_code == 200, resp.text
    payload = resp.json()

    # Response fields
    assert "dataset_id" in payload and isinstance(payload["dataset_id"], int)
    assert payload["name"] == "tickets.csv"
    assert payload["row_count"] == 3
    assert payload["department_count"] == 3  # IT, HR, Finance
    assert "file_hash" in payload and isinstance(payload["file_hash"], str) and payload["file_hash"]
    assert payload.get("inserted_tickets") == 3

    dataset_id = payload["dataset_id"]

    # Verify persistence via repositories
    db = SessionLocal()
    try:
        # Using repository to recompute ensures dataset exists and updates department_count
        count = DatasetsRepository.recompute_department_count(db, dataset_id)
        assert count == 3

        tickets = TicketsRepository.query_filtered(db, dataset_id=dataset_id, limit=100)
        assert len(tickets) == 3
        # Spot-check ticket fields normalization
        departments = sorted({t.department for t in tickets if t.department})
        assert departments == ["Finance", "HR", "IT"]
        assignment_groups = sorted({t.assignment_group for t in tickets if t.assignment_group})
        assert "Service Desk" in assignment_groups and "Onboarding" in assignment_groups and "Accounts" in assignment_groups
        products = sorted({t.product for t in tickets if t.product})
        assert "Laptop" in products and "HRIS" in products and "ERP" in products
    finally:
        db.close()

    # Re-upload identical file; should not duplicate tickets and return same dataset_id
    resp2 = client.post("/ingest/upload", files=files)
    assert resp2.status_code == 200
    payload2 = resp2.json()
    assert payload2["dataset_id"] == dataset_id
    assert payload2["inserted_tickets"] == 0  # idempotent guard