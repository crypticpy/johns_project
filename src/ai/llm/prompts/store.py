from __future__ import annotations

import json
import re
from pathlib import Path

TEMPLATES_DIR = Path(__file__).resolve().parent / "prompts" / "templates"
META_SUFFIX = ".meta.json"
CONTENT_SUFFIX = ".md"


def _ensure_templates_dir() -> None:
    """
    Ensure the templates directory exists. No-op if already present.
    """
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)


def _safe_version_name(version: str) -> str:
    """
    Sanitize the version string to a filesystem-safe name.

    - Allow alphanumerics, dots, dashes, underscores.
    - Collapse invalid characters to '-'.
    - Trim leading/trailing separators.
    """
    s = (version or "").strip()
    if not s:
        raise ValueError("version is required")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s)
    s = s.strip("-._")
    if not s:
        raise ValueError("version resolves to an empty filename after sanitization")
    return s


def _content_path(version: str) -> Path:
    return TEMPLATES_DIR / f"{_safe_version_name(version)}{CONTENT_SUFFIX}"


def _meta_path(version: str) -> Path:
    return TEMPLATES_DIR / f"{_safe_version_name(version)}{META_SUFFIX}"


def list_versions() -> list[str]:
    """
    List available prompt template versions (file-based).

    Returns:
        Sorted list of version names (without suffix), e.g., ["v1", "experiment-2024.10"].
    """
    _ensure_templates_dir()
    versions: list[str] = []
    for p in TEMPLATES_DIR.glob(f"*{CONTENT_SUFFIX}"):
        name = p.name
        if name.endswith(CONTENT_SUFFIX):
            base = name[: -len(CONTENT_SUFFIX)]
            if base:
                versions.append(base)
    versions.sort(key=lambda x: x.lower())
    return versions


def load_template(version: str) -> str:
    """
    Load the template content for the given version.

    Raises:
        FileNotFoundError if the template does not exist.
    """
    _ensure_templates_dir()
    path = _content_path(version)
    if not path.exists():
        raise FileNotFoundError(f"Template version not found: {version}")
    return path.read_text(encoding="utf-8")


def save_template(version: str, metadata: dict[str, object] | None = None) -> dict[str, object]:
    """
    Save a template and optional metadata.

    Behavior:
    - If metadata contains a 'template' key (str), writes it to {version}.md.
    - Writes metadata (without the 'template' content) to {version}.meta.json.
    - Creates the templates directory if missing.
    - Path handling is sanitized to prevent directory traversal.

    Args:
        version: template version identifier (e.g., 'v1', 'onboarding-2024.10')
        metadata: optional dict; if includes 'template' (str), it is used as content.

    Returns:
        Dict with {'version': str, 'content_written': bool, 'meta_written': bool}
    """
    _ensure_templates_dir()
    content_written = False
    meta_written = False

    content: str | None = None
    if metadata and isinstance(metadata.get("template"), str):
        content = str(metadata["template"])

    content_path = _content_path(version)
    if content is not None:
        content_path.write_text(content, encoding="utf-8")
        content_written = True

    # Write metadata sidecar without the large 'template' body
    meta = (metadata or {}).copy()
    if "template" in meta:
        del meta["template"]
    meta["version"] = _safe_version_name(version)

    meta_path = _meta_path(version)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_written = True

    return {"version": version, "content_written": content_written, "meta_written": meta_written}
