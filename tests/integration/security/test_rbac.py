from __future__ import annotations

import os
import time
from typing import Dict

import jwt
from fastapi.testclient import TestClient
from sqlalchemy import select

from api.main import create_app
from config.settings import get_settings
from db.models import Dataset
from db.session import SessionLocal, init_db


def _setup_env_for_rbac(secret: str = "devsecret") -> None:
    os.environ["APP_ENABLE_RBAC"] = "true"
    os.environ["APP_JWT_SECRET"] = secret
    # Ensure offline analyzer to avoid network during CI
    os.environ["ANALYZER_BACKEND"] = "offline"


def _create_dataset() -> int:
    init_db()
    with SessionLocal() as db:
        # Unique file_hash each run
        fh = f"testhash-{int(time.time() * 1000)}"
        ds = Dataset(name="rbac-test", file_hash=fh, row_count=0, department_count=0)
        db.add(ds)
        db.commit()
        db.refresh(ds)
        return int(ds.id)


def _make_token(secret: str, roles: Dict[str, str | list[str] | None], subject: str | None = None) -> str:
    payload: Dict[str, object] = {}
    if subject:
        payload["sub"] = subject
    # Flexible roles payload
    for key, val in roles.items():
        if val is not None:
            payload[key] = val
    return jwt.encode(payload, secret, algorithm="HS256")  # type: ignore[arg-type]


def test_analysis_run_requires_token_and_roles():
    _setup_env_for_rbac()
    # Clear cached settings to read new env
    get_settings.cache_clear()  # type: ignore[attr-defined]

    app = create_app()
    client = TestClient(app)

    dataset_id = _create_dataset()

    # No token -> 403
    resp1 = client.post(
        "/analysis/run",
        json={"dataset_id": dataset_id, "question": "Q", "analyzer_backend": "offline"},
    )
    assert resp1.status_code == 403

    # Token with analyst role -> 200
    token_ok = _make_token("devsecret", roles={"roles": ["analyst"]}, subject="tester")
    resp2 = client.post(
        "/analysis/run",
        headers={"Authorization": f"Bearer {token_ok}"},
        json={"dataset_id": dataset_id, "question": "Q", "analyzer_backend": "offline"},
    )
    assert resp2.status_code == 200

    # Token missing required role -> 403
    token_bad = _make_token("devsecret", roles={"roles": ["viewer"]}, subject="tester")
    resp3 = client.post(
        "/analysis/run",
        headers={"Authorization": f"Bearer {token_bad}"},
        json={"dataset_id": dataset_id, "question": "Q", "analyzer_backend": "offline"},
    )
    assert resp3.status_code == 403