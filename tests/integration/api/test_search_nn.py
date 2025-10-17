from __future__ import annotations

import io
from typing import Any, Dict, List, Tuple

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


def test_search_nn_returns_neighbors_and_fields_and_filters_and_is_idempotent():
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

    # Run kNN search with a relevant query
    k = 5
    resp_search = client.get(
        "/search/nn",
        params={
            "dataset_id": dataset_id,
            "q": "login problem with account",
            "k": k,
        },
    )
    assert resp_search.status_code == 200, resp_search.text
    payload_search: Dict[str, Any] = resp_search.json()

    assert payload_search["dataset_id"] == dataset_id
    assert int(payload_search["k"]) == k
    assert payload_search["model_name"] in ("builtin-384", "text-embedding-3-large")  # model_name inferred from meta; builtin expected
    results: List[Dict[str, Any]] = payload_search.get("results", [])
    assert len(results) <= k
    # Validate fields in results
    for item in results:
        assert "ticket_id" in item and isinstance(item["ticket_id"], int)
        assert "score" in item and isinstance(item["score"], float)
        assert "department" in item
        assert "product" in item
        assert "summary" in item

    # Department filter: only "IT" should appear when requested
    resp_search_dept = client.get(
        "/search/nn",
        params={
            "dataset_id": dataset_id,
            "q": "login account issue",
            "k": k,
            "department": "IT",
        },
    )
    assert resp_search_dept.status_code == 200, resp_search_dept.text
    payload_search_dept = resp_search_dept.json()
    results_dept: List[Dict[str, Any]] = payload_search_dept.get("results", [])
    for item in results_dept:
        assert item["department"] == "IT"

    # Idempotency: re-run /embed/run and /search/nn and assert stable outputs
    resp_embed_again = client.post("/embed/run", json=req_body)
    assert resp_embed_again.status_code == 200, resp_embed_again.text

    resp_search_again = client.get(
        "/search/nn",
        params={
            "dataset_id": dataset_id,
            "q": "login problem with account",
            "k": k,
        },
    )
    assert resp_search_again.status_code == 200, resp_search_again.text
    payload_search_again = resp_search_again.json()
    results_again: List[Dict[str, Any]] = payload_search_again.get("results", [])

    # Compare by (ticket_id, rounded score) sequences for stability
    def sig(seq: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
        return [(int(x["ticket_id"]), round(float(x["score"]), 6)) for x in seq]

    assert sig(results_again) == sig(results)