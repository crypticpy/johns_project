from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def _safe_series(df: pd.DataFrame, column: str) -> pd.Series | None:
    """
    Return a Series for the given column if present, else None.
    Ensures return type is pandas Series to leverage vectorized ops.
    """
    if not isinstance(df, pd.DataFrame):
        return None
    if column not in df.columns:
        return None
    return df[column]


def _clean_str_series(s: pd.Series) -> pd.Series:
    """
    Normalize a string-like series:
    - Drop NaNs
    - Convert values to strings and strip whitespace
    - Filter out empty strings after stripping
    """
    s = s.dropna()
    s = s.map(lambda x: "" if x is None else str(x).strip())
    s = s[s != ""]
    return s


def compute_quality_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """
    Compute distribution for ticket quality labels.

    Column: "ticket_quality"

    Fallback: if column missing or empty after cleanup, return {}.
    """
    s = _safe_series(df, "ticket_quality")
    if s is None:
        return {}
    s = _clean_str_series(s)
    if s.empty:
        return {}
    counts = s.value_counts(dropna=True)
    # Cast counts to int for JSON stability
    return {str(k): int(v) for k, v in counts.to_dict().items()}


def compute_complexity_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """
    Compute distribution for ticket resolution complexity labels.

    Column: "resolution_complexity"

    Fallback: if column missing or empty after cleanup, return {}.
    """
    s = _safe_series(df, "resolution_complexity")
    if s is None:
        return {}
    s = _clean_str_series(s)
    if s.empty:
        return {}
    counts = s.value_counts(dropna=True)
    return {str(k): int(v) for k, v in counts.to_dict().items()}


def compute_department_volume(df: pd.DataFrame, top_n: int = 10) -> List[tuple[str, int]]:
    """
    Compute top-N department volumes as a sorted list of (department, count).

    Column: "Department"
    Ordering: by count desc, then department asc for deterministic ties.

    Fallback: if column missing or empty after cleanup, return [].
    """
    s = _safe_series(df, "Department")
    if s is None:
        return []
    s = _clean_str_series(s)
    if s.empty:
        return []
    counts = s.value_counts(dropna=True)
    # Build list and sort deterministically
    items = [(str(k), int(v)) for k, v in counts.to_dict().items()]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    if top_n is not None and top_n > 0:
        items = items[: int(top_n)]
    return items


def compute_reassignment_distribution(df: pd.DataFrame) -> Dict[int, int]:
    """
    Compute distribution for reassignment group counts.

    Column: "Reassignment group count tracking_index"
    Behavior:
      - Coerce to numeric; drop NaNs; cast to int
      - Return dict[int, int] mapping reassignment_count -> frequency
      - Sorted by key naturally when converted to dict (order not guaranteed, but deterministic from pandas v2)

    Fallback: if column missing or empty after cleanup, return {}.
    """
    s = _safe_series(df, "Reassignment group count tracking_index")
    if s is None:
        return {}
    # Coerce to numeric, ignore non-numeric gracefully
    s_num = pd.to_numeric(s, errors="coerce").dropna()
    if s_num.empty:
        return {}
    s_int = s_num.astype("int64")
    counts = s_int.value_counts(dropna=True).sort_index()
    # Ensure Python ints for keys and values
    return {int(k): int(v) for k, v in counts.to_dict().items()}


def compute_product_distribution(df: pd.DataFrame) -> List[tuple[str, int]]:
    """
    Compute product distribution as a list of (product, count) across all products.

    Column: "extract_product"
    Ordering: by count desc, then product asc for deterministic ties.

    Fallback: if column missing or empty after cleanup, return [].
    """
    s = _safe_series(df, "extract_product")
    if s is None:
        return []
    s = _clean_str_series(s)
    if s.empty:
        return []
    counts = s.value_counts(dropna=True)
    items = [(str(k), int(v)) for k, v in counts.to_dict().items()]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return items