from __future__ import annotations

import io
from typing import Dict, Any

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from db.models import Base
from db.session import engine


def _make_full_csv_bytes() -> bytes:
    # Canonical and variant columns to exercise normalization
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


def _make_reduced_csv_bytes() -> bytes:
    # Reduced CSV missing several canonical columns (fallback behavior)
    df = pd.DataFrame(
        [
            {
                "Department": "IT",
                "Summary": "Login issue",
            },
            {
                "Department": "HR",
                "Summary": "Onboarding question",
            },
        ]
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def test_analytics_metrics_returns_distributions_and_counts_align_with_uploaded_data():
    # Fresh DB
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    app = create_app()
    client = TestClient(app)

    # Upload dataset with full canonical coverage
    files = {"file": ("tickets_full.csv", _make_full_csv_bytes(), "text/csv")}
    resp_ingest = client.post("/ingest/upload", files=files)
    assert resp_ingest.status_code == 200, resp_ingest.text
    payload_ingest = resp_ingest.json()
    dataset_id = int(payload_ingest["dataset_id"])
    assert payload_ingest["row_count"] == 3

    # Call analytics metrics
    resp_metrics = client.get("/analytics/metrics", params={"dataset_id": dataset_id, "top_n": 10})
    assert resp_metrics.status_code == 200, resp_metrics.text
    metrics_payload: Dict[str, Any] = resp_metrics.json()
    assert metrics_payload["dataset_id"] == dataset_id
    assert "metrics" in metrics_payload and isinstance(metrics_payload["metrics"], dict)

    m = metrics_payload["metrics"]

    # Quality distribution: good, average, poor each 1
    assert m["quality"]["type"] == "bar"
    assert set(m["quality"]["labels"]) == {"good", "average", "poor"}
    assert sorted(m["quality"]["values"]) == [1, 1, 1]

    # Complexity distribution: low, medium, high each 1
    assert m["complexity"]["type"] == "bar"
    assert set(m["complexity"]["labels"]) == {"low", "medium", "high"}
    assert sorted(m["complexity"]["values"]) == [1, 1, 1]

    # Department volume: IT, HR, Finance
    assert m["department_volume"]["type"] == "bar"
    assert set(m["department_volume"]["labels"]) == {"IT", "HR", "Finance"}
    assert sorted(m["department_volume"]["values"]) == [1, 1, 1]

    # Reassignment histogram: 0, 2, 3 buckets
    assert m["reassignment"]["type"] == "histogram"
    assert set(m["reassignment"]["buckets"]) == {0, 2, 3}
    assert sorted(m["reassignment"]["counts"]) == [1, 1, 1]

    # Product distribution: Laptop, HRIS, ERP
    assert m["product"]["type"] == "bar"
    assert set(m["product"]["labels"]) == {"Laptop", "HRIS", "ERP"}
    assert sorted(m["product"]["values"]) == [1, 1, 1]


def test_analytics_metrics_missing_columns_fallback_returns_empty_structures():
    # Fresh DB
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    app = create_app()
    client = TestClient(app)

    # Upload reduced dataset missing quality/complexity/product/reassignment
    files = {"file": ("tickets_reduced.csv", _make_reduced_csv_bytes(), "text/csv")}
    resp_ingest = client.post("/ingest/upload", files=files)
    assert resp_ingest.status_code == 200, resp_ingest.text
    payload_ingest = resp_ingest.json()
    dataset_id = int(payload_ingest["dataset_id"])
    assert payload_ingest["row_count"] == 2

    # Call analytics metrics
    resp_metrics = client.get("/analytics/metrics", params={"dataset_id": dataset_id, "top_n": 10})
    assert resp_metrics.status_code == 200, resp_metrics.text
    metrics_payload: Dict[str, Any] = resp_metrics.json()
    assert metrics_payload["dataset_id"] == dataset_id
    m = metrics_payload["metrics"]

    # Department should have data
    assert m["department_volume"]["type"] == "bar"
    assert set(m["department_volume"]["labels"]) == {"IT", "HR"}
    assert sorted(m["department_volume"]["values"]) == [1, 1]

    # Missing columns yield empty structures
    assert m["quality"]["labels"] == [] and m["quality"]["values"] == []
    assert m["complexity"]["labels"] == [] and m["complexity"]["values"] == []
    assert m["product"]["labels"] == [] and m["product"]["values"] == []
    assert m["reassignment"]["buckets"] == [] and m["reassignment"]["counts"] == []