from __future__ import annotations

from typing import Dict, List, Tuple, Any


def _to_bar_spec(dist: Dict[str, int], title: str) -> Dict[str, Any]:
    """
    Convert a label->count mapping into a minimal chart-agnostic bar spec.
    Deterministic ordering: sort by count desc, then label asc.
    """
    if not dist:
        return {"type": "bar", "title": title, "labels": [], "values": []}

    items = [(str(k), int(v)) for k, v in dist.items()]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    return {"type": "bar", "title": title, "labels": labels, "values": values}


def _to_ranked_list_spec(pairs: List[Tuple[str, int]], title: str) -> Dict[str, Any]:
    """
    Convert a ranked list like department/product volumes [(label, count), ...]
    into a minimal chart spec (bar).
    Assumes the input list is already sorted deterministically by the caller.
    """
    if not pairs:
        return {"type": "bar", "title": title, "labels": [], "values": []}
    labels = [str(k) for k, _ in pairs]
    values = [int(v) for _, v in pairs]
    return {"type": "bar", "title": title, "labels": labels, "values": values}


def _to_histogram_spec(dist: Dict[int, int], title: str) -> Dict[str, Any]:
    """
    Convert an integer value->count mapping into a histogram-like spec.
    Deterministic ordering by bucket (ascending).
    """
    if not dist:
        return {"type": "histogram", "title": title, "buckets": [], "counts": []}
    items = [(int(k), int(v)) for k, v in dist.items()]
    items.sort(key=lambda kv: kv[0])
    buckets = [k for k, _ in items]
    counts = [v for _, v in items]
    return {"type": "histogram", "title": title, "buckets": buckets, "counts": counts}


def transform_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform raw analytics metrics into chart-agnostic specifications.

    Expected input keys:
      - quality: Dict[str, int]
      - complexity: Dict[str, int]
      - department_volume: List[Tuple[str, int]]
      - reassignment: Dict[int, int]
      - product: List[Tuple[str, int]]

    Returns:
      {
        "quality": {type, title, labels, values},
        "complexity": {type, title, labels, values},
        "department_volume": {type, title, labels, values},
        "reassignment": {type, title, buckets, counts},
        "product": {type, title, labels, values},
      }
    """
    quality = metrics.get("quality") or {}
    complexity = metrics.get("complexity") or {}
    dept_vol = metrics.get("department_volume") or []
    reassignment = metrics.get("reassignment") or {}
    product = metrics.get("product") or []

    return {
        "quality": _to_bar_spec(quality, title="Ticket Quality Distribution"),
        "complexity": _to_bar_spec(complexity, title="Resolution Complexity Distribution"),
        "department_volume": _to_ranked_list_spec(dept_vol, title="Top Departments by Volume"),
        "reassignment": _to_histogram_spec(reassignment, title="Reassignment Count Distribution"),
        "product": _to_ranked_list_spec(product, title="Top Products by Volume"),
    }