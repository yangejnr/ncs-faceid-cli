from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import faiss


@dataclass(frozen=True)
class Match:
    person_id: int
    score: float


class FaceIndex:
    """
    Cosine similarity via inner product on L2-normalized vectors.
    """
    def __init__(self, vectors: np.ndarray, person_ids: List[int]):
        self.person_ids = person_ids
        if vectors.size == 0:
            self.dim = 0
            self.index = None
            return

        vecs = np.asarray(vectors, dtype=np.float32)
        self.dim = vecs.shape[1]
        # Ensure unit norm
        faiss.normalize_L2(vecs)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)

    def search(self, query: np.ndarray, top_k: int = 3) -> List[Match]:
        if self.index is None:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)
        out: List[Match] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            out.append(Match(person_id=self.person_ids[idx], score=float(score)))
        return out
