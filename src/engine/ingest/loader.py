from __future__ import annotations

import hashlib
from io import BytesIO

import pandas as pd

from engine.preprocess.clean import normalize_columns


def _md5_hex(data: bytes) -> str:
    h = hashlib.md5()
    h.update(data)
    return h.hexdigest()


def _read_tabular(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Read CSV or Excel into a DataFrame from in-memory bytes.
    Pure function: no filesystem access.
    """
    name = filename.lower().strip()
    bio = BytesIO(file_bytes)

    if name.endswith(".csv") or name.endswith(".txt"):
        # Let pandas infer delimiter; users typically provide CSV
        return pd.read_csv(bio)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        # Prefer openpyxl for .xlsx; pandas will choose engine when possible
        try:
            return pd.read_excel(bio)
        except ImportError as e:  # pragma: no cover - depends on environment extras
            raise ValueError(
                "Excel support requires 'openpyxl' to be installed. "
                "Install openpyxl or upload a CSV file."
            ) from e

    # Fallback: attempt CSV; error bubbles up with informative message
    try:
        bio.seek(0)
        return pd.read_csv(bio)
    except Exception as e:
        raise ValueError(f"Unsupported file type for '{filename}'. Provide CSV or Excel.") from e


def validate_and_load(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, dict]:
    """
    Validate and load an uploaded file into a normalized DataFrame and metadata.

    Returns:
        (df, metadata) where metadata includes:
          - file_hash: md5 hex of raw bytes
          - rows: number of rows after load (before filtering)
    """
    if not isinstance(file_bytes, (bytes, bytearray)):
        raise TypeError("file_bytes must be bytes")

    file_hash = _md5_hex(file_bytes)

    df = _read_tabular(file_bytes, filename)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Loaded object is not a DataFrame")

    # Normalize columns to canonical schema
    df = normalize_columns(df)

    metadata = {
        "file_hash": file_hash,
        "rows": int(len(df)),
        "filename": filename,
    }
    return df, metadata
