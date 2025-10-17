from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ai.rerank.interface import RerankAdapter


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """
    Deterministic, simple regex tokenization:
    - Alphanumeric word tokens
    - Lowercased
    - No stemming or stopword removal (keeps offline determinism)
    """
    if not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(text)]


def _tf(tokens: List[str]) -> Dict[str, int]:
    """Compute term frequencies deterministically."""
    freq: Dict[str, int] = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0) + 1
    return freq


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Standard Jaccard similarity between two sets, normalized to [0, 1]."""
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return inter / union


def _weighted_jaccard(tf_a: Dict[str, int], tf_b: Dict[str, int]) -> float:
    """
    Weighted Jaccard over term frequencies:
    sum(min(tf_a[t], tf_b[t])) / sum(max(tf_a[t], tf_b[t]))
    """
    if not tf_a and not tf_b:
        return 0.0
    keys = set(tf_a.keys()) | set(tf_b.keys())
    num = 0
    den = 0
    for k in keys:
        a = tf_a.get(k, 0)
        b = tf_b.get(k, 0)
        num += min(a, b)
        den += max(a, b)
    if den == 0:
        return 0.0
    return num / den


class BuiltinLexicalReranker(RerankAdapter):
    """
    Deterministic offline lexical reranker.

    Scoring:
      - Tokenize query and candidate summary via simple regex
      - Compute:
          jaccard = |tokens_query ∩ tokens_doc| / |tokens_query ∪ tokens_doc|
          wj = weighted_jaccard over term frequencies
      - Combine:
          score_raw = 0.5 * jaccard + 0.5 * wj
      - Normalize strictly to [0.0, 1.0] (already bounded, but clamp for safety)
    Ordering:
      - Sort by score desc
      - Stable tiebreak by ascending ticket_id to maintain deterministic order across runs
    """

    def rerank(self, query: str, candidates: List[Tuple[int, str]]) -> List[Tuple[int, float]]:
        q_tokens = _tokenize(query or "")
        q_tf = _tf(q_tokens)
        q_set = set(q_tokens)

        out: List[Tuple[int, float]] = []
        for ticket_id, summary in candidates:
            s_tokens = _tokenize(summary or "")
            s_tf = _tf(s_tokens)
            s_set = set(s_tokens)

            j = _jaccard(q_set, s_set)
            wj = _weighted_jaccard(q_tf, s_tf)

            score = 0.5 * j + 0.5 * wj
            # Clamp to [0, 1] for robustness
            if score < 0.0:
                score = 0.0
            elif score > 1.0:
                score = 1.0

            out.append((int(ticket_id), float(score)))

        # Deterministic ordering: score desc, then ticket_id asc
        out.sort(key=lambda x: (-x[1], x[0]))
        return out


__all__ = ["BuiltinLexicalReranker"]