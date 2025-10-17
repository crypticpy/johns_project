from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def _to_2d_array(vectors: list[list[float]]) -> np.ndarray:
    """
    Convert list[list[float]] to a 2D numpy array (float32) with basic validation.
    Pure utility: no external side effects.
    """
    if not isinstance(vectors, list) or not vectors:
        raise ValueError("vectors must be a non-empty list of lists")
    try:
        arr = np.asarray(vectors, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Failed to convert vectors to array: {e}") from e
    if arr.ndim != 2:
        raise ValueError("vectors must be a 2D list (list of lists of floats)")
    return arr


def kmeans_cluster(
    vectors: list[list[float]],
    n_clusters: int,
    random_state: int = 42,
) -> dict[str, object]:
    """
    K-Means clustering over embedding vectors.

    Returns dict with keys:
      - "assignments": list[int] per vector index
      - "centroids": list[list[float]] cluster centers
      - "silhouette": float or None
    """
    if not isinstance(n_clusters, int) or n_clusters <= 0:
        # Invalid parameter; API should map to 400
        raise ValueError("n_clusters must be a positive integer")
    if n_clusters == 1:
        # Explicit guard for spec; silhouette undefined for single cluster
        # API should map to 400 invalid params
        raise ValueError("n_clusters must be greater than 1 for kmeans")

    X = _to_2d_array(vectors)
    if X.shape[0] < n_clusters:
        # Insufficient samples; API should map to 422
        raise ValueError("insufficient vectors for requested number of clusters")

    # Deterministic behavior via fixed random_state and explicit n_init
    km = KMeans(n_clusters=n_clusters, random_state=int(random_state), n_init=10)
    km.fit(X)

    labels: list[int] = [int(x) for x in km.labels_.tolist()]
    centroids: list[list[float]] = [[float(v) for v in row.tolist()] for row in km.cluster_centers_]

    # Silhouette: only when valid (≥2 clusters and enough samples)
    sil: float | None = None
    unique_labels = sorted(set(labels))
    if len(unique_labels) >= 2 and X.shape[0] > n_clusters:
        try:
            sil = float(silhouette_score(X, labels, metric="euclidean"))
        except Exception:
            sil = None

    return {
        "assignments": labels,
        "centroids": centroids,
        "silhouette": sil,
    }


def hdbscan_cluster(
    vectors: list[list[float]],
    min_cluster_size: int = 5,
    min_samples: int = 5,
) -> dict[str, object]:
    """
    HDBSCAN clustering over embedding vectors.

    Returns dict with keys:
      - "assignments": list[int] (noise as -1)
      - "silhouette": float or None (computed on non-noise points when ≥2 clusters)

    Notes:
    - This function requires the optional 'hdbscan' package. If unavailable, it raises RuntimeError.
    - Engine remains pure; API must guard availability and map errors to HTTP 422.
    """
    try:
        import hdbscan  # type: ignore  # pylint: disable=import-error
    except Exception as e:
        raise RuntimeError(f"hdbscan dependency not available: {e}") from e

    if not isinstance(min_cluster_size, int) or min_cluster_size <= 1:
        raise ValueError("min_cluster_size must be an integer greater than 1")
    if not isinstance(min_samples, int) or min_samples <= 0:
        raise ValueError("min_samples must be a positive integer")

    X = _to_2d_array(vectors)
    if X.shape[0] < max(min_cluster_size, 2):
        raise ValueError("insufficient vectors for HDBSCAN")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size), min_samples=int(min_samples)
    )
    labels_np = clusterer.fit_predict(X)
    labels: list[int] = [int(x) for x in labels_np.tolist()]

    # Silhouette on non-noise points when there are ≥2 clusters
    sil: float | None = None
    mask = labels_np >= 0
    if np.any(mask):
        labeled = labels_np[mask]
        unique = sorted(set(int(x) for x in labeled.tolist()))
        if len(unique) >= 2 and int(mask.sum()) >= 3:
            try:
                sil = float(silhouette_score(X[mask], labeled.tolist(), metric="euclidean"))
            except Exception:
                sil = None

    return {
        "assignments": labels,
        "silhouette": sil,
    }


def tfidf_top_terms(
    texts: list[str],
    assignments: list[int],
    top_k: int = 10,
) -> dict[int, list[tuple[str, float]]]:
    """
    Compute TF-IDF top terms per cluster (cluster_id ≥ 0). Noise (-1) is skipped.

    Returns:
      dict[cluster_id -> list[(term, score)]]

    Determinism:
    - Vocabulary derived from all texts with fixed preprocessor/token pattern.
    - Per-cluster terms sorted by score desc, then term asc; truncated to top_k.
    """
    if (
        not isinstance(texts, list)
        or not isinstance(assignments, list)
        or len(texts) != len(assignments)
    ):
        raise ValueError("texts and assignments must be lists of equal length")

    n = len(texts)
    if n == 0:
        return {}

    # Fit vectorizer on all texts to maintain consistent vocabulary across clusters
    # Avoid stop_words to keep domain-specific tokens
    vec = TfidfVectorizer(lowercase=True)
    try:
        tfidf = vec.fit_transform([t if isinstance(t, str) else "" for t in texts])
    except Exception as e:
        raise ValueError(f"Failed to compute TF-IDF: {e}") from e

    feature_names = vec.get_feature_names_out()
    # Build per-cluster aggregated term weights
    by_cluster: dict[int, list[int]] = {}
    for idx, cid in enumerate(assignments):
        if cid is None or int(cid) < 0:
            continue  # skip noise
        cid_int = int(cid)
        by_cluster.setdefault(cid_int, []).append(idx)

    result: dict[int, list[tuple[str, float]]] = {}
    for cid, row_indices in by_cluster.items():
        if not row_indices:
            result[cid] = []
            continue
        # Sum TF-IDF scores across rows belonging to this cluster
        sub = tfidf[row_indices, :]
        # Convert to dense sum vector efficiently
        sums = np.asarray(sub.sum(axis=0)).ravel()
        # Get indices of non-zero features
        nonzero_idx = np.where(sums > 0)[0]
        terms_scores: list[tuple[str, float]] = []
        for j in nonzero_idx:
            term = str(feature_names[j])
            score = float(sums[j])
            terms_scores.append((term, score))
        # Sort deterministically: score desc, term asc
        terms_scores.sort(key=lambda ts: (-ts[1], ts[0]))
        if top_k is not None and top_k > 0:
            terms_scores = terms_scores[: int(top_k)]
        result[cid] = terms_scores

    return result
