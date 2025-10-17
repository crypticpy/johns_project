from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd

from engine.analytics.metrics import (
    compute_complexity_distribution,
    compute_department_volume,
    compute_quality_distribution,
)


@dataclass(frozen=True)
class SamplingConfig:
    """
    Configuration for stratified sampling.

    Fields:
    - max_tickets: hard cap on total sampled tickets across all segments
    - token_budget: approximate token budget for context_text; tokens ~= chars/4
    - buckets: list of column names to form Cartesian segments; missing columns are ignored
    - per_bucket_cap: maximum tickets to sample per Cartesian segment
    """
    max_tickets: int
    token_budget: int
    buckets: List[str] = field(default_factory=lambda: ["Department", "ticket_quality", "resolution_complexity"])
    per_bucket_cap: int = 3


def _existing_bucket_cols(df: pd.DataFrame, names: List[str]) -> List[str]:
    return [c for c in (names or []) if isinstance(df, pd.DataFrame) and c in df.columns]


def _safe_len(obj: Any) -> int:
    try:
        return int(len(obj))  # type: ignore[arg-type]
    except Exception:
        return 0


def _approx_chars_for_tokens(tokens: int) -> int:
    """
    Approximate tokens as chars/4, invert to char budget.
    """
    t = int(tokens) if tokens is not None else 0
    if t <= 0:
        return 10_000_000  # effectively unlimited for this operation
    return t * 4


def _row_text(rec: Dict[str, Any]) -> str:
    """
    Build a concise, sanitized line for a ticket record using available fields.
    Preference order for text: summarize_ticket, normalized_text, Summary/summary.
    Include Department and extract_product if present to aid LLM grounding.
    """
    dept = str(rec.get("Department") or "").strip()
    prod = str(rec.get("extract_product") or "").strip()
    text = (
        rec.get("summarize_ticket")
        or rec.get("normalized_text")
        or rec.get("Summary")
        or rec.get("summary")
        or ""
    )
    s = " ".join(str(text).split())  # collapse whitespace
    parts: List[str] = []
    if dept:
        parts.append(f"[Dept: {dept}]")
    if prod:
        parts.append(f"[Prod: {prod}]")
    if s:
        parts.append(s)
    line = " ".join(parts).strip()
    if not line:
        line = "(no text)"
    return line


def _resolve_id_column(df: pd.DataFrame) -> str | None:
    """
    Prefer explicit ticket id column if present; otherwise None (use index).
    """
    for cand in ("id", "ticket_id", "ticketId"):
        if cand in df.columns:
            return cand
    return None


def stratified_sample(df: pd.DataFrame, cfg: SamplingConfig) -> Dict[str, Any]:
    """
    Stratified sample over Cartesian buckets with deterministic selection and token-budgeted context.

    Returns dict with:
      - "sampled_ids": list[int]
      - "segments": list[dict] with keys: {"key": {bucket: value, ...}, "count": int, "sample_count": int}
      - "summary": dict with aggregate metrics (for prompt conditioning)
      - "context_text": str bounded by token_budget (trimmed at segment boundaries)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("stratified_sample requires a pandas DataFrame")

    # Identify bucket columns that exist in the DataFrame
    bucket_cols = _existing_bucket_cols(df, cfg.buckets or [])
    if not bucket_cols:
        # Treat entire DataFrame as a single segment
        bucket_cols = []

    # Construct deterministic group keys
    if bucket_cols:
        # Replace NaN with explicit placeholder for grouping stability
        group_df = df.copy()
        for col in bucket_cols:
            if col in group_df.columns:
                group_df[col] = group_df[col].fillna("(missing)").astype(str)
        grouped = group_df.groupby(bucket_cols, dropna=False, sort=True)
        groups: List[Tuple[Tuple[str, ...], pd.DataFrame]] = []
        for key_vals, gdf in grouped:
            # Ensure key is always a tuple for uniform handling
            if not isinstance(key_vals, tuple):
                key_vals = (key_vals,)
            groups.append((tuple(key_vals), gdf))
        # Deterministic ordering by key (lexicographic)
        groups.sort(key=lambda kv: tuple(str(x) for x in kv[0]))
    else:
        groups = [(tuple(), df)]

    # Determine id column if available
    id_col = _resolve_id_column(df)

    sampled_rows: List[pd.Series] = []
    segments_out: List[Dict[str, Any]] = []

    remaining_capacity = max(int(cfg.max_tickets or 0), 0)
    per_cap = max(int(cfg.per_bucket_cap or 0), 0)

    for key_vals, gdf in groups:
        if remaining_capacity <= 0:
            break
        seg_total = len(gdf)
        # Determine how many to sample from this segment
        take_n = min(seg_total, per_cap if per_cap > 0 else seg_total, remaining_capacity)
        if take_n <= 0:
            # Still record the segment for visibility
            seg_key_map = {bucket_cols[i]: key_vals[i] for i in range(len(key_vals))}
            segments_out.append({"key": seg_key_map, "count": int(seg_total), "sample_count": 0})
            continue

        # Deterministic selection: random_state fixed, but final ordering stable by original index
        gdf_shuffled = gdf.sample(n=take_n, random_state=42) if take_n < seg_total else gdf
        chosen = gdf_shuffled.copy()
        # Stabilize order by original index
        chosen = chosen.sort_index()
        for _, row in chosen.iterrows():
            sampled_rows.append(row)

        remaining_capacity -= take_n
        seg_key_map = {bucket_cols[i]: key_vals[i] for i in range(len(key_vals))}
        segments_out.append({"key": seg_key_map, "count": int(seg_total), "sample_count": int(take_n)})

    # Build sampled ids in stable order
    sampled_ids: List[int] = []
    for row in sampled_rows:
        if id_col and id_col in row and pd.notna(row[id_col]):
            try:
                sampled_ids.append(int(row[id_col]))  # explicit ticket id
                continue
            except Exception:
                pass
        # Fallback to DataFrame index (cast to int if possible)
        try:
            sampled_ids.append(int(row.name))  # type: ignore[arg-type]
        except Exception:
            # Last resort: enumerate
            sampled_ids.append(len(sampled_ids))

    # Summary metrics for prompts (from full DataFrame to reflect dataset context)
    summary: Dict[str, Any] = {
        "total_rows": int(len(df)),
        "selected_rows": int(len(sampled_rows)),
        "quality": compute_quality_distribution(df),
        "complexity": compute_complexity_distribution(df),
        "top_departments": compute_department_volume(df, top_n=5),
        "bucket_columns": bucket_cols,
    }

    # Build token-budgeted context text
    # Assemble segment blocks with small headers plus bullet samples
    char_budget = _approx_chars_for_tokens(int(cfg.token_budget or 0))
    header_lines: List[str] = [
        "# Analysis Context",
        f"Total tickets: {summary['total_rows']}",
        f"Selected for sampling: {summary['selected_rows']}",
    ]

    # Add compact metrics lines
    if summary.get("top_departments"):
        tops = ", ".join([f"{d}({c})" for d, c in summary["top_departments"][:5]])
        header_lines.append(f"Top Departments: {tops}")
    if summary.get("quality"):
        q_items = ", ".join([f"{k}({v})" for k, v in sorted(summary["quality"].items())])
        header_lines.append(f"Quality: {q_items}")
    if summary.get("complexity"):
        c_items = ", ".join([f"{k}({v})" for k, v in sorted(summary["complexity"].items())])
        header_lines.append(f"Complexity: {c_items}")

    blocks: List[str] = ["\n".join(header_lines)]

    # Map from segment to its sampled rows for block assembly
    # Build a DataFrame from sampled_rows to efficiently filter per segment (if buckets exist)
    sampled_df = pd.DataFrame(sampled_rows) if sampled_rows else pd.DataFrame(columns=df.columns)
    if bucket_cols and not sampled_df.empty:
        # Ensure same placeholder normalization for comparison
        for col in bucket_cols:
            if col in sampled_df.columns:
                sampled_df[col] = sampled_df[col].fillna("(missing)").astype(str)

    # Build blocks per segment following segments_out order
    for seg in segments_out:
        if int(seg.get("sample_count", 0)) <= 0:
            # Still include header for visibility; no samples
            key_map = seg.get("key", {})
            seg_title_parts = [f"{k}={v}" for k, v in key_map.items()]
            block_lines = [f"## Segment: {' | '.join(seg_title_parts) if seg_title_parts else 'All'} "
                           f"(count={seg.get('count', 0)}, sampled=0)"]
            seg_block = "\n".join(block_lines)
            # Budget check at boundary: include whole block or stop
            if sum(len(b) for b in blocks) + len(seg_block) + 1 > char_budget:
                break
            blocks.append(seg_block)
            continue

        key_map = seg.get("key", {})
        seg_title_parts = [f"{k}={v}" for k, v in key_map.items()]
        header = f"## Segment: {' | '.join(seg_title_parts) if seg_title_parts else 'All'} " \
                 f"(count={seg.get('count', 0)}, sampled={seg.get('sample_count', 0)})"

        samples_lines: List[str] = []
        # Select rows for this segment
        if bucket_cols and not sampled_df.empty:
            mask = pd.Series([True] * len(sampled_df))
            for col, val in key_map.items():
                if col in sampled_df.columns:
                    mask = mask & (sampled_df[col].astype(str) == str(val))
            seg_rows = sampled_df[mask]
        else:
            # No buckets; all rows belong to one segment
            seg_rows = sampled_df

        # Build bullet lines for each row in deterministic order
        for _, rec in seg_rows.sort_index().iterrows():
            line = _row_text(rec.to_dict())
            samples_lines.append(f"- {line}")

        seg_block = "\n".join([header] + samples_lines)
        # Apply token/char budget at segment boundary
        if sum(len(b) for b in blocks) + len(seg_block) + 1 > char_budget:
            # Do not include partial segment; stop here
            break
        blocks.append(seg_block)

    context_text = "\n\n".join(blocks).strip()

    return {
        "sampled_ids": sampled_ids,
        "segments": segments_out,
        "summary": summary,
        "context_text": context_text,
    }