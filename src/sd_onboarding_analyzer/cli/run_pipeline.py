from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import jwt  # PyJWT
from fastapi import FastAPI
from fastapi.testclient import TestClient
from config.settings import get_settings


@dataclass(frozen=True)
class EmbedConfig:
    backend: str = "builtin"
    model_name: str = "builtin-384"


def _detect_content_type(filename: str) -> str:
    """
    Detect a suitable content type for upload based on file extension.
    """
    name = filename.lower().strip()
    if name.endswith(".csv") or name.endswith(".txt"):
        return "text/csv"
    if name.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if name.endswith(".xls"):
        return "application/vnd.ms-excel"
    # fallback to octet-stream to avoid incorrect parsers
    return "application/octet-stream"


def _raise_for_non_200(resp) -> None:
    """
    Raise a RuntimeError with readable details if HTTP response is non-200.
    """
    if resp.status_code != 200:
        # Prefer JSON detail when available
        try:
            payload = resp.json()
            detail = payload.get("detail") or payload
            raise RuntimeError(f"HTTP {resp.status_code}: {detail}")
        except Exception:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")


def run_ingest(app: FastAPI, file_path: str, dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Ingest a CSV/XLSX file via POST /ingest/upload.

    Args:
        app: FastAPI application
        file_path: path to a local CSV or Excel file
        dataset_name: optional dataset name to override filename on server

    Returns:
        dict payload: {
            "dataset_id", "name", "row_count", "department_count",
            "file_hash", "inserted_tickets"
        }

    Raises:
        FileNotFoundError: when file_path is not found
        RuntimeError: for HTTP or parsing errors
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)
    content_type = _detect_content_type(filename)

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    client = TestClient(app)
    files = {"file": (dataset_name or filename, file_bytes, content_type)}
    resp = client.post("/ingest/upload", files=files)
    _raise_for_non_200(resp)
    payload = resp.json()
    # Ensure expected keys exist
    for k in ("dataset_id", "row_count"):
        if k not in payload:
            raise RuntimeError(f"Ingest response missing '{k}'")
    return payload


def run_embed(
    app: FastAPI,
    dataset_id: int,
    backend: str = EmbedConfig.backend,
    model_name: str = EmbedConfig.model_name,
) -> Dict[str, Any]:
    """
    Run embeddings and FAISS indexing via POST /embed/run.

    Args:
        app: FastAPI application
        dataset_id: dataset identifier
        backend: 'builtin' or 'sentence-transformers' (defaults to offline/builtin)
        model_name: embedder model name (defaults to 'builtin-384')

    Returns:
        dict payload: {
            "dataset_id", "model_name", "backend",
            "vector_dim", "embedded_count", "indexed"
        }

    Raises:
        RuntimeError: for HTTP or logical errors
    """
    body = {
        "dataset_id": int(dataset_id),
        "backend": backend,
        "model_name": model_name,
    }
    client = TestClient(app)
    resp = client.post("/embed/run", json=body)
    _raise_for_non_200(resp)
    payload = resp.json()
    for k in ("dataset_id", "embedded_count", "indexed"):
        if k not in payload:
            raise RuntimeError(f"Embed response missing '{k}'")
    return payload


def run_analysis(
    app: FastAPI,
    dataset_id: int,
    question: str,
    prompt_version: str = "v1",
    max_tickets: int = 50,
    token_budget: int = 8000,
) -> Dict[str, Any]:
    """
    Execute analysis via POST /analysis/run using offline analyzer by default.

    If RBAC is enabled via settings (APP_ENABLE_RBAC=true), attach a minimal
    HS256 JWT with 'analyst' role to satisfy guards in offline CI runs.

    Args:
        app: FastAPI application
        dataset_id: dataset identifier
        question: analysis question
        prompt_version: prompt template version (default: 'v1')
        max_tickets: sampling cap (default: 50)
        token_budget: token budget for context construction (default: 8000)

    Returns:
        dict payload: {
            "analysis_id", "dataset_id",
            "prompt_version", "ticket_count", "created_at"
        }

    Raises:
        RuntimeError: for HTTP or logical errors
    """
    body = {
        "dataset_id": int(dataset_id),
        "question": question,
        "prompt_version": prompt_version,
        # Force offline determinism explicitly; env defaults also applied by app_factory
        "analyzer_backend": "offline",
        "max_tickets": int(max_tickets),
        "token_budget": int(token_budget),
    }

    # Always attach a minimal JWT with analyst role to satisfy RBAC if enabled.
    # Safe when RBAC disabled (header is ignored).
    headers: Dict[str, str] = {}
    try:
        settings = get_settings()
    except Exception:
        settings = None  # type: ignore[assignment]

    secret = (
        os.getenv("APP_JWT_SECRET")
        or (getattr(settings, "jwt_secret", None) if settings is not None else None)
        or "devsecret"
    )
    try:
        token = jwt.encode({"sub": "cli", "roles": ["analyst"]}, secret, algorithm="HS256")  # type: ignore[arg-type]
        headers["Authorization"] = f"Bearer {token}"
    except Exception:
        # If PyJWT is unavailable for some reason, proceed without header; tests will surface issues.
        headers = {}

    client = TestClient(app)
    resp = client.post("/analysis/run", json=body, headers=headers or None)
    _raise_for_non_200(resp)
    payload = resp.json()
    for k in ("analysis_id", "dataset_id"):
        if k not in payload:
            raise RuntimeError(f"Analysis response missing '{k}'")
    return payload


def run_report(app: FastAPI, dataset_id: int) -> Dict[str, Any]:
    """
    Retrieve report markdown via GET /reports/{dataset_id}.

    Args:
        app: FastAPI application
        dataset_id: dataset identifier

    Returns:
        dict payload: {
            "dataset_id", "report_markdown", "analysis_count"
        }

    Raises:
        RuntimeError: for HTTP or logical errors
    """
    client = TestClient(app)
    url = f"/reports/{int(dataset_id)}"
    resp = client.get(url)
    _raise_for_non_200(resp)
    payload = resp.json()
    for k in ("dataset_id", "report_markdown"):
        if k not in payload:
            raise RuntimeError(f"Report response missing '{k}'")
    return payload


def run_pipeline(
    app: FastAPI,
    file_path: str,
    question: str,
    prompt_version: str = "v1",
    *,
    embed_backend: str = EmbedConfig.backend,
    embed_model: str = EmbedConfig.model_name,
    max_tickets: int = 50,
    token_budget: int = 8000,
) -> Dict[str, Any]:
    """
    Orchestrate end-to-end pipeline: ingest → embed → analysis → report.

    Args:
        app: FastAPI application
        file_path: CSV/XLSX input path
        question: analysis question
        prompt_version: prompt version (default 'v1')
        embed_backend: embeddings backend (default 'builtin')
        embed_model: model name (default 'builtin-384')
        max_tickets: sampling cap (default 50)
        token_budget: token budget (default 8000)

    Returns:
        dict payload: {
            "dataset_id", "analysis_id", "report_markdown",
            "ingest": {...}, "embed": {...}, "analysis": {...}, "report": {...}
        }

    Raises:
        FileNotFoundError: when file_path not found
        RuntimeError: for HTTP or logical errors
    """
    ingest_res = run_ingest(app, file_path)
    dataset_id = int(ingest_res["dataset_id"])

    embed_res = run_embed(app, dataset_id, backend=embed_backend, model_name=embed_model)

    analysis_res = run_analysis(
        app,
        dataset_id,
        question=question,
        prompt_version=prompt_version,
        max_tickets=max_tickets,
        token_budget=token_budget,
    )
    analysis_id = int(analysis_res["analysis_id"])

    report_res = run_report(app, dataset_id)

    return {
        "dataset_id": dataset_id,
        "analysis_id": analysis_id,
        "report_markdown": report_res.get("report_markdown", ""),
        "ingest": ingest_res,
        "embed": embed_res,
        "analysis": analysis_res,
        "report": report_res,
    }


def to_json(data: Dict[str, Any]) -> str:
    """
    Serialize dict payloads to compact JSON for CLI printing.
    """
    return json.dumps(data, ensure_ascii=False)