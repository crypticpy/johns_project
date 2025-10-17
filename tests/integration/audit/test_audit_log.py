from __future__ import annotations

import os
import time

import jwt
from fastapi.testclient import TestClient
from sqlalchemy import select

from api.main import create_app
from config.settings import get_settings
from db.models import AuditLog, Dataset
from db.session import SessionLocal, init_db


def _setup_env_for_rbac(secret: str = "devsecret") -> None:
    os.environ["APP_ENABLE_RBAC"] = "true"
    os.environ["APP_JWT_SECRET"] = secret
    # Ensure offline analyzer to avoid any network during CI
    os.environ["ANALYZER_BACKEND"] = "offline"


def _create_dataset() -> int:
    init_db()
    with SessionLocal() as db:
        fh = f"audit-hash-{int(time.time() * 1000)}"
        ds = Dataset(name="audit-test", file_hash=fh, row_count=0, department_count=0)
        db.add(ds)
        db.commit()
        db.refresh(ds)
        return int(ds.id)


def _make_token(secret: str, subject: str) -> str:
    payload: dict[str, object] = {"sub": subject, "roles": ["analyst"]}
    return jwt.encode(payload, secret, algorithm="HS256")  # type: ignore[arg-type]


def test_analysis_run_emits_audit_log_entry():
    _setup_env_for_rbac()
    # Clear cached settings to read new env values
    get_settings.cache_clear()  # type: ignore[attr-defined]

    app = create_app()
    client = TestClient(app)

    dataset_id = _create_dataset()
    token = _make_token("devsecret", subject="auditor")

    # Run analysis offline
    resp = client.post(
        "/analysis/run",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "dataset_id": dataset_id,
            "question": "What changed?",
            "analyzer_backend": "offline",
            "prompt_version": "v1",
            "max_tickets": 10,
            "token_budget": 1000,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    analysis_id = int(payload["analysis_id"])

    # Query audit log for the emitted entry
    with SessionLocal() as db:
        rows = list(db.execute(select(AuditLog).order_by(AuditLog.id.desc()).limit(25)).scalars())
        # Find matching record
        target = None
        for r in rows:
            if r.action == "analysis.run" and r.resource == f"dataset:{dataset_id}":
                target = r
                break

        assert target is not None, "Expected an AuditLog entry for analysis.run"
        assert target.subject == "auditor"
        assert target.action == "analysis.run"
        assert target.resource == f"dataset:{dataset_id}"
        meta = getattr(target, "metadata_", None) or {}
        assert int(meta.get("analysis_id")) == analysis_id
        assert str(meta.get("prompt_version")) == "v1"
