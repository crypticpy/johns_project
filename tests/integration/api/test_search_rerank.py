from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from db.models import Base
from db.session import engine


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


def test_search_rerank_builtin_changes_order_when_lexical_overlap_differs_and_respects_filters() -> (
    None
):
    # Ensure a clean database for this test run (idempotent)
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

    # Build embeddings and FAISS index using builtin backend for offline determinism
    req_body = {
        "dataset_id": dataset_id,
        "backend": "builtin",
        "model_name": "builtin-384",
        "batch_size": 32,
    }
    resp_embed = client.post("/embed/run", json=req_body)
    assert resp_embed.status_code == 200, resp_embed.text
    payload_embed = resp_embed.json()
    assert payload_embed["indexed"] is True
    assert payload_embed["model_name"] == "builtin-384"

    # Run kNN search with rerank enabled (builtin backend) for two different queries.
    # Query A should favor the "login" ticket lexically; Query B should favor the "account" ticket lexically.
    resp_rerank_a = client.get(
        "/search/nn",
        params={
            "dataset_id": dataset_id,
            "q": "login issue with account",
            "k": 5,
            "rerank": "true",
            "rerank_backend": "builtin",
        },
    )
    assert resp_rerank_a.status_code == 200, resp_rerank_a.text
    payload_a: dict[str, Any] = resp_rerank_a.json()
    results_a: list[dict[str, Any]] = payload_a.get("results", [])
    assert len(results_a) > 0
    # Validate normalized scores
    for item in results_a:
        s = float(item["score"])
        assert 0.0 <= s <= 1.0

    top_a = int(results_a[0]["ticket_id"])

    resp_rerank_b = client.get(
        "/search/nn",
        params={
            "dataset_id": dataset_id,
            "q": "new employee account provisioning",
            "k": 5,
            "rerank": "true",
            "rerank_backend": "builtin",
        },
    )
    assert resp_rerank_b.status_code == 200, resp_rerank_b.text
    payload_b: dict[str, Any] = resp_rerank_b.json()
    results_b: list[dict[str, Any]] = payload_b.get("results", [])
    assert len(results_b) > 0
    for item in results_b:
        s = float(item["score"])
        assert 0.0 <= s <= 1.0

    top_b = int(results_b[0]["ticket_id"])

    # Expect reranked top items to differ across queries due to lexical overlap differences
    assert top_a != top_b

    # Department filter scenario: only "HR" should appear when requested
    resp_rerank_hr = client.get(
        "/search/nn",
        params={
            "dataset_id": dataset_id,
            "q": "account setup for new employee",
            "k": 5,
            "department": "HR",
            "rerank": "true",
            "rerank_backend": "builtin",
        },
    )
    assert resp_rerank_hr.status_code == 200, resp_rerank_hr.text
    payload_hr = resp_rerank_hr.json()
    results_hr: list[dict[str, Any]] = payload_hr.get("results", [])
    for item in results_hr:
        assert item["department"] == "HR"
