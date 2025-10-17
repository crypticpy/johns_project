from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _venv_python() -> str:
    """
    Resolve Python interpreter from project virtual environment (.venv) if present.
    Falls back to current interpreter when .venv is missing.
    """
    root = Path(__file__).resolve().parents[3]
    # macOS/Linux venv layout
    vpy = root / ".venv" / "bin" / "python"
    if vpy.exists():
        return str(vpy)
    # Windows venv layout
    vpy_win = root / ".venv" / "Scripts" / "python.exe"
    if vpy_win.exists():
        return str(vpy_win)
    return sys.executable


def _write_sample_csv(path: Path) -> None:
    """
    Create a minimal CSV compatible with the ingest normalization rules.
    """
    path.write_text(
        "\n".join(
            [
                "Department,Summary,Quality,Complexity,Reassignment group count tracking_index",
                "IT,User cannot login,High,Medium,2",
                "HR,Onboarding paperwork missing,Medium,Low,1",
                "Finance,Expense approval delayed,Low,High,3",
            ]
        ),
        encoding="utf-8",
    )


def _offline_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """
    Build environment ensuring offline determinism for embeddings and analyzer.
    """
    env = dict(base or os.environ)
    env.setdefault("APP_EMBED_BACKEND", "builtin")
    env.setdefault("ANALYZER_BACKEND", "offline")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    return env


@pytest.mark.integration
def test_cli_pipeline_module_run(tmp_path: Path) -> None:
    """
    Invoke the CLI via Python module path to avoid PATH issues.

    Command:
      python -m sd_onboarding_analyzer.cli.__main__ pipeline --file {tmp_csv} --question "Test"
    """
    # Ensure dev database directory exists for SQLite
    (Path.cwd() / "data").mkdir(parents=True, exist_ok=True)

    csv_path = tmp_path / "tickets.csv"
    _write_sample_csv(csv_path)

    cmd = [
        _venv_python(),
        "-m",
        "sd_onboarding_analyzer.cli.__main__",
        "pipeline",
        "--file",
        str(csv_path),
        "--question",
        "Test",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(Path.cwd()),
        env=_offline_env(),
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, f"CLI failed: {proc.stderr}"
    assert proc.stdout, "CLI produced no output"
    out = proc.stdout.strip()

    # The pipeline prints final report markdown unless --out is provided
    # Accept either report header or analytics snapshot section
    assert (
        "# Dataset Report" in out
        or "## Current Analytics Snapshot" in out
        or "analysis_id" in out
    ), f"Unexpected CLI output:\n{out}"


@pytest.mark.integration
def test_run_pipeline_direct_helpers(tmp_path: Path) -> None:
    """
    Call run_pipeline() directly for structure assertions without subprocess overhead.
    """
    # Ensure dev database directory exists for SQLite
    (Path.cwd() / "data").mkdir(parents=True, exist_ok=True)

    csv_path = tmp_path / "tickets.csv"
    _write_sample_csv(csv_path)

    # Import locally to ensure package resolution without relying on PATH
    from sd_onboarding_analyzer.cli.app_factory import get_app  # noqa: WPS433
    from sd_onboarding_analyzer.cli.run_pipeline import run_pipeline  # noqa: WPS433

    app = get_app()

    result = run_pipeline(
        app,
        file_path=str(csv_path),
        question="Test",
        prompt_version="v1",
        embed_backend="builtin",
        embed_model="builtin-384",
        max_tickets=50,
        token_budget=8000,
    )

    assert isinstance(result, dict), "run_pipeline did not return a dict"
    for key in ("dataset_id", "analysis_id", "report_markdown", "ingest", "embed", "analysis", "report"):
        assert key in result, f"Missing key in pipeline result: {key}"
    assert isinstance(result["dataset_id"], int)
    assert isinstance(result["analysis_id"], int)
    assert isinstance(result["report_markdown"], str)
    assert result["report_markdown"], "Empty report_markdown"
    assert "## Current Analytics Snapshot" in result["report_markdown"] or "# Dataset Report" in result["report_markdown"]