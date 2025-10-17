from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from db.models import Base
from db.repositories.embeddings_repo import EmbeddingsRepository
from db.session import SessionLocal, engine


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


def test_embed_run_persists_embeddings_and_builds_faiss_index_offline():
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

    # Run embeddings using builtin backend to avoid any network
    req_body = {
        "dataset_id": dataset_id,
        "backend": "builtin",
        "model_name": "builtin-384",
        "batch_size": 32,
    }
    resp_embed = client.post("/embed/run", json=req_body)
    assert resp_embed.status_code == 200, resp_embed.text
    payload_embed = resp_embed.json()

    assert payload_embed["dataset_id"] == dataset_id
    assert payload_embed["backend"] == "builtin"
    assert payload_embed["model_name"] == "builtin-384"
    assert payload_embed["embedded_count"] == row_count
    assert payload_embed["indexed"] is True
    vector_dim = int(payload_embed["vector_dim"])
    assert vector_dim in (256, 384, 512)
    assert vector_dim == 384

    # Assert FAISS index files exist
    index_dir = Path("data/faiss")
    index_path = index_dir / f"{dataset_id}.index"
    meta_path = index_dir / f"{dataset_id}.meta.json"
    assert index_path.exists()
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert int(meta.get("dim", 0)) == vector_dim
    assert meta.get("model_name") == "builtin-384"

    # Re-run embeddings to ensure idempotency (no duplicate rows persisted)
    resp_embed_again = client.post("/embed/run", json=req_body)
    assert resp_embed_again.status_code == 200, resp_embed_again.text
    payload_embed_again = resp_embed_again.json()
    # Same dataset/model; should still succeed and keep index durable
    assert payload_embed_again["indexed"] is True
    assert int(payload_embed_again["vector_dim"]) == vector_dim

    # Verify repository state: embeddings exist for dataset+model and count == row_count
    db = SessionLocal()
    try:
        assert EmbeddingsRepository.exists_for_dataset(db, dataset_id, "builtin-384") is True
        ids_all, vecs_all = EmbeddingsRepository.fetch_by_dataset(db, dataset_id, "builtin-384")
        assert len(ids_all) == row_count
        assert len(vecs_all) == row_count
        # Confirm dimension
        assert len(vecs_all[0]) == vector_dim
    finally:
        db.close()
