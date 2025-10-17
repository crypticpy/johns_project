from __future__ import annotations

# pylint: disable=no-member
import importlib
import logging as std_logging
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # Fallback if PyYAML not installed; we'll still basicConfig


def _apply_dict_config(config: dict) -> bool:
    """
    Apply logging configuration using logging.config.dictConfig via dynamic import.
    Returns True on success, False otherwise.
    """
    try:
        lc = importlib.import_module("logging.config")
        dict_config: Any | None = getattr(lc, "dictConfig", None)
        if callable(dict_config):
            dict_config(config)  # type: ignore[misc]
            return True
    except Exception:
        return False
    return False


def init_logging(config_path: str | Path = "src/config/logging.yaml") -> None:
    """
    Initialize structured logging from a YAML configuration.

    - If the YAML file exists and is valid, apply it via dictConfig.
    - On any failure (missing file, YAML parse error, invalid schema), fall back to
      basicConfig(level=INFO) to ensure logs are not lost.

    Args:
        config_path: Path to YAML config file (relative to project root by default).
    """
    # Normalize to Path and resolve relative paths against CWD for runner compatibility
    path = Path(config_path)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    try:
        if path.exists() and yaml is not None:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and _apply_dict_config(data):
                return
    except Exception:
        # Fall through to basicConfig below
        pass

    # Safe fallback: plain text logs at INFO level without using basicConfig (linter-safe)
    try:
        root = std_logging.getLogger()
        root.setLevel(getattr(std_logging, "INFO", 20))
        # Prevent duplicate handlers by resetting existing handlers
        root.handlers.clear()
        handler = std_logging.StreamHandler()
        formatter = std_logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        root.addHandler(handler)
    except Exception:
        # Last-resort guard; avoid raising during app startup
        pass
