from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

# pylint: disable=E1120,no-value-for-parameter


class FaissIndexError(RuntimeError):
    """Raised for FAISS index load/build/search failures."""


def _norm(v: np.ndarray) -> np.ndarray:
    """L2-normalize vectors; guard against zero vectors."""
    denom = np.linalg.norm(v, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return v / denom


def _should_disable_faiss() -> bool:
    """
    Determine if FAISS should be disabled.

    Default: disabled (safer for offline/CI to avoid native segfaults).
    Enable only if an explicit flag is set truthy:
      - APP_FAISS_ENABLED
      - FAISS_ENABLED
    """
    for key in ("APP_FAISS_ENABLED", "FAISS_ENABLED"):
        val = os.environ.get(key)
        if val and str(val).strip().lower() in ("1", "true", "yes", "on"):
            return False
    return True


def _try_import_faiss() -> Any | None:
    """
    Lazily import faiss. Return module if available and not disabled, else None.
    Never raise at import time to preserve offline/CI stability.
    """
    if _should_disable_faiss():
        return None
    try:
        import faiss  # type: ignore
    except Exception:
        return None
    return faiss


class FaissIndexAdapter:
    """
    FAISS vector index adapter with per-dataset persistence and safe numpy fallback.

    Storage layout:
      - data/faiss/{dataset_id}.index       (FAISS index file or placeholder for fallback)
      - data/faiss/{dataset_id}.meta.json   (metadata: {"dim": int, "model_name": str})
      - data/faiss/{dataset_id}.npz         (numpy fallback store: {"vectors": float32[n, d], "ids": int64[n]})

    Similarity:
      - Uses Inner Product (IP) with unit-normalized embeddings -> cosine similarity.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = (base_dir or Path("data/faiss")).resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _index_path(self, dataset_id: int) -> Path:
        return self._base_dir / f"{int(dataset_id)}.index"

    def _meta_path(self, dataset_id: int) -> Path:
        return self._base_dir / f"{int(dataset_id)}.meta.json"

    def _npz_path(self, dataset_id: int) -> Path:
        return self._base_dir / f"{int(dataset_id)}.npz"

    def _write_meta(self, dataset_id: int, dim: int, model_name: str) -> None:
        meta = {"dim": int(dim), "model_name": str(model_name)}
        path = self._meta_path(dataset_id)
        path.write_text(json.dumps(meta), encoding="utf-8")

    def _read_meta(self, dataset_id: int) -> dict | None:
        path = self._meta_path(dataset_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    # ---- Numpy fallback persistence helpers ----

    def _save_npz(self, dataset_id: int, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Persist fallback store and ensure .index placeholder exists."""
        npz_path = self._npz_path(dataset_id)
        np.savez_compressed(npz_path, vectors=vectors.astype(np.float32), ids=ids.astype(np.int64))
        # Touch .index file for compatibility with existing tests
        idx_path = self._index_path(dataset_id)
        try:
            if not idx_path.exists():
                idx_path.touch()
        except Exception:
            # Best-effort; absence is non-fatal for search logic
            pass

    def _load_npz(self, dataset_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        npz_path = self._npz_path(dataset_id)
        if not npz_path.exists():
            return None
        try:
            with np.load(npz_path) as data:
                vectors = np.asarray(data["vectors"], dtype=np.float32)
                ids = np.asarray(data["ids"], dtype=np.int64)
                return vectors, ids
        except Exception:
            return None

    # ---- FAISS helpers (lazy) ----

    def _create_faiss_index(self, dim: int) -> Any:
        faiss = _try_import_faiss()
        if faiss is None:
            raise FaissIndexError("FAISS disabled or unavailable")
        base = faiss.IndexFlatIP(dim)
        return faiss.IndexIDMap2(base)

    def _save_faiss_index(self, index: Any, path: Path) -> None:
        faiss = _try_import_faiss()
        if faiss is None:
            raise FaissIndexError("FAISS disabled or unavailable")
        faiss.write_index(index, str(path))

    def _load_faiss_index(self, dataset_id: int) -> Any | None:
        faiss = _try_import_faiss()
        if faiss is None:
            return None
        path = self._index_path(dataset_id)
        if not path.exists():
            return None
        try:
            return faiss.read_index(str(path))
        except Exception as e:
            raise FaissIndexError(
                f"Failed to load FAISS index for dataset {dataset_id}: {e}"
            ) from e

    # ---- Public interface ----

    def build_index(
        self,
        dataset_id: int,
        vectors: list[list[float]],
        ids: list[int],
        model_name: str | None = None,
    ) -> None:
        """
        Build (or rebuild) the index for a dataset from scratch, overwriting existing index.

        Writes index and metadata atomically (best-effort).
        """
        if not vectors or not ids or len(vectors) != len(ids):
            raise FaissIndexError("build_index requires equal-length non-empty vectors and ids")

        dim = len(vectors[0])
        # Validate consistent dimensions
        for v in vectors:
            if len(v) != dim:
                raise FaissIndexError("All vectors must share the same dimension")

        faiss_mod = _try_import_faiss()

        xb = np.asarray(vectors, dtype=np.float32)
        xids = np.asarray(ids, dtype=np.int64)

        if faiss_mod is not None:
            # Use FAISS when available
            index = self._create_faiss_index(dim)
            try:
                index.add_with_ids(xb, xids)
            except Exception as e:
                raise FaissIndexError(f"Failed to add vectors to FAISS index: {e}") from e
            # Persist index
            idx_path = self._index_path(dataset_id)
            self._save_faiss_index(index, idx_path)
        else:
            # Fallback: persist numpy arrays and touch placeholder .index
            self._save_npz(dataset_id, xb, xids)

        # Persist metadata
        self._write_meta(dataset_id, dim, model_name or "")

    def add(self, dataset_id: int, vectors: list[list[float]], ids: list[int]) -> None:
        """
        Add vectors/ids to an existing index; creates a new index if none exists.

        Ensures dimensionality matches existing index; if mismatch, rebuild required by caller.
        """
        if not vectors or not ids or len(vectors) != len(ids):
            raise FaissIndexError("add requires equal-length non-empty vectors and ids")

        dim = len(vectors[0])
        for v in vectors:
            if len(v) != dim:
                raise FaissIndexError("All vectors must share the same dimension")

        faiss_mod = _try_import_faiss()
        xb = np.asarray(vectors, dtype=np.float32)
        xids = np.asarray(ids, dtype=np.int64)

        if faiss_mod is not None:
            index = self._load_faiss_index(dataset_id)
            if index is None:
                # No index exists: build it
                self.build_index(dataset_id, vectors, ids)
                return
            # Check dimensionality compatibility
            try:
                existing_dim = index.d
            except Exception:
                existing_dim = dim
            if existing_dim != dim:
                raise FaissIndexError(
                    f"Vector dimension mismatch: index dim={existing_dim}, vectors dim={dim}"
                )

            try:
                if hasattr(index, "add_with_ids"):
                    index.add_with_ids(xb, xids)
                else:
                    idmap = faiss_mod.IndexIDMap2(index)
                    idmap.add_with_ids(xb, xids)
                    index = idmap
            except Exception as e:
                raise FaissIndexError(f"Failed to add vectors to FAISS index: {e}") from e

            # Persist updated index
            self._save_faiss_index(index, self._index_path(dataset_id))
        else:
            # Fallback: merge into numpy store
            loaded = self._load_npz(dataset_id)
            if loaded is None:
                self._save_npz(dataset_id, xb, xids)
                return
            vecs_old, ids_old = loaded
            if vecs_old.shape[1] != dim:
                raise FaissIndexError(
                    f"Vector dimension mismatch: index dim={vecs_old.shape[1]}, vectors dim={dim}"
                )
            vecs_new = np.concatenate([vecs_old, xb], axis=0)
            ids_new = np.concatenate([ids_old, xids], axis=0)
            self._save_npz(dataset_id, vecs_new, ids_new)

    def search(self, dataset_id: int, vector: list[float], k: int) -> list[tuple[int, float]]:
        """
        Search the dataset index for nearest neighbors; returns [(id, score)].

        Requires that the index exists; raises if missing.
        """
        faiss_mod = _try_import_faiss()

        if faiss_mod is not None:
            index = self._load_faiss_index(dataset_id)
            if index is None:
                raise FaissIndexError(f"Index for dataset {dataset_id} does not exist")

            dim = len(vector)
            if getattr(index, "d", dim) != dim:
                raise FaissIndexError(
                    f"Query vector dimension {dim} does not match index dimension {getattr(index, 'd', '?')}"
                )

            xq = np.asarray([vector], dtype=np.float32)
            try:
                distances, neighbors = index.search(xq, int(k))
            except Exception as e:
                raise FaissIndexError(f"FAISS search failed: {e}") from e

            # Extract result arrays
            dists = distances[0]
            nns = neighbors[0]
            out: list[tuple[int, float]] = []
            for i in range(len(nns)):
                nid = int(nns[i])
                if nid == -1:
                    continue
                out.append((nid, float(dists[i])))
            return out

        # Fallback: numpy cosine search on persisted store
        loaded = self._load_npz(dataset_id)
        if loaded is None:
            raise FaissIndexError(f"Index for dataset {dataset_id} does not exist")
        vecs, ids = loaded
        if vecs.size == 0:
            return []

        q = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != vecs.shape[1]:
            raise FaissIndexError(
                f"Query vector dimension {q.shape[1]} does not match index dimension {vecs.shape[1]}"
            )

        # Normalize to compute cosine similarity via dot product
        vecs_n = _norm(vecs)
        q_n = _norm(q)
        scores = (vecs_n @ q_n.T).reshape(-1)  # [n]

        # Top-k indices by score desc
        k = int(max(1, k))
        if k >= scores.shape[0]:
            top_idx = np.argsort(-scores)
        else:
            # argpartition for efficiency then sort within top-k
            top_k_unsorted = np.argpartition(-scores, k - 1)[:k]
            top_idx = top_k_unsorted[np.argsort(-scores[top_k_unsorted])]

        results: list[tuple[int, float]] = []
        for i in top_idx:
            results.append((int(ids[i]), float(scores[i])))
        return results
