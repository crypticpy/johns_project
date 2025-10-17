from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

import pandas as pd


EMAIL_PATTERN = re.compile(
    r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"
)
PHONE_PATTERN = re.compile(
    r"(?x)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}"
)


def clean_text(s: str, *, lowercase: bool = False) -> str:
    """
    Basic text cleanup:
    - strip leading/trailing whitespace
    - normalize inner whitespace to single spaces
    - optional lowercase
    """
    s = s.strip()
    # Replace multiple whitespace (including tabs/newlines) with single space
    s = re.sub(r"\s+", " ", s)
    if lowercase:
        s = s.lower()
    return s


def redact_pii(s: str) -> str:
    """
    Redact simple PII patterns:
    - emails → [REDACTED_EMAIL]
    - US phone numbers → [REDACTED_PHONE]
    """
    s = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", s)
    s = PHONE_PATTERN.sub("[REDACTED_PHONE]", s)
    return s


def _first_available(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """
    Return the first candidate column name present in df (exact match), else None.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize incoming DataFrame columns to canonical names used across charts and analysis:

    Canonical targets:
      - "Department"
      - "Assignment Group"
      - "extract_product"
      - "summarize_ticket"
      - "ticket_quality"
      - "resolution_complexity"
      - "Reassignment group count tracking_index"

    Strategy:
      - Merge variant columns into the canonical one if both exist
      - Else rename a present variant to the canonical name
      - Else create the canonical column with None values
    """
    # Work on a copy to keep function pure
    df_norm = df.copy()

    def ensure_column(canon: str, variants: List[str]) -> None:
        """
        Merge/rename variants into a canonical column. If the canonical already exists,
        fill its NaN values from any variant columns, then drop the variants.
        If the canonical doesn't exist, rename the first present variant to canonical
        and merge/fill from any additional variants, dropping them afterwards.
        If none exist, create the canonical column with None.
        """
        # Determine which variants are present (exact header match)
        present_variants = [v for v in variants if v in df_norm.columns]

        if canon in df_norm.columns:
            # Fill missing values from variants and drop them
            for alt in present_variants:
                if alt == canon:
                    continue
                df_norm[canon] = df_norm[canon].fillna(df_norm[alt])
                df_norm.drop(columns=[alt], inplace=True)
        else:
            if present_variants:
                chosen = present_variants[0]
                if chosen != canon:
                    df_norm.rename(columns={chosen: canon}, inplace=True)
                # After potential rename, merge remaining variants
                for alt in present_variants[1:]:
                    if alt == canon:
                        continue
                    df_norm[canon] = df_norm[canon].fillna(df_norm[alt])
                    df_norm.drop(columns=[alt], inplace=True)
            else:
                # Create empty canonical column
                df_norm[canon] = None

    # Apply normalization/merging for each canonical field
    ensure_column(
        "Department",
        ["Department", "department", "Dept", "dept", "DEPARTMENT"],
    )
    ensure_column(
        "Assignment Group",
        ["Assignment Group", "Assignment_Group", "assignment group", "assignment_group", "ASSIGNMENT GROUP"],
    )
    ensure_column(
        "extract_product",
        ["extract_product", "Product", "product", "PRODUCT", "Category", "Service"],
    )
    ensure_column(
        "summarize_ticket",
        ["summarize_ticket", "Summary", "Short description", "short description", "Description", "description"],
    )
    ensure_column(
        "ticket_quality",
        ["ticket_quality", "Quality", "quality", "QUALITY", "information_completeness"],
    )
    ensure_column(
        "resolution_complexity",
        ["resolution_complexity", "Resolution Complexity", "resolution complexity", "Complexity", "complexity", "historical_similarity"],
    )
    ensure_column(
        "Reassignment group count tracking_index",
        [
            "Reassignment group count tracking_index",
            "Reassignment_group_count_tracking_index",
            "Reassignment Count",
            "reassignment_count",
            "Reassignment count",
            "Reassignment",
        ],
    )

    return df_norm