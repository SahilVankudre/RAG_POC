"""FAISS implementation of the ``VectorIndex`` port.

Uses exact inner-product search (``IndexFlatIP``). Because stored and query vectors are
L2-normalized, the inner product equals cosine similarity. Exact search over 10K-50K vectors of
384 dimensions takes milliseconds, so an approximate index would add complexity for no gain.
"""

from pathlib import Path

import faiss
import numpy as np


class FaissVectorIndex:
    """Wraps a FAISS flat inner-product index; positions are insertion order."""

    def __init__(self, index: faiss.Index) -> None:
        self._index = index

    @classmethod
    def from_vectors(cls, vectors: np.ndarray) -> "FaissVectorIndex":
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        return cls(index)

    @classmethod
    def read(cls, location: Path) -> "FaissVectorIndex":
        return cls(faiss.read_index(str(location)))

    @property
    def dimension(self) -> int:
        return int(self._index.d)

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    @property
    def index_type(self) -> str:
        return type(self._index).__name__

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0 or self.size == 0:
            return []
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, self.dimension)
        scores, positions = self._index.search(query, min(top_k, self.size))
        return [
            (int(position), float(score))
            for position, score in zip(positions[0], scores[0], strict=True)
            if position >= 0  # FAISS pads with -1 when fewer results exist
        ]

    def write(self, location: Path) -> None:
        location.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(location))
