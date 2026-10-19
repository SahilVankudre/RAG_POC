from collections.abc import Sequence

import numpy as np
import pytest

from rag_poc.embedding import embed_in_batches, fingerprint_embedding_input
from rag_poc.lib import EmbeddingError


class FakeEmbedder:
    """Deterministic 3-d unit vectors derived from each text; records batch sizes."""

    model_name = "fake-model"
    dimension = 3

    def __init__(self, *, normalize: bool = True) -> None:
        self.batch_sizes: list[int] = []
        self._normalize = normalize

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.batch_sizes.append(len(texts))
        vectors = np.array([[len(text), ord(text[0]), 1.0] for text in texts], dtype=np.float32)
        if self._normalize:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


def test_rows_stay_aligned_with_input_order_despite_length_sorted_batches() -> None:
    texts = ["a much longer review text", "ok", "medium review", "z", "another long-ish one"]
    embedder = FakeEmbedder()
    progress: list[tuple[int, int]] = []

    vectors = embed_in_batches(
        embedder,
        texts,
        batch_size=2,
        on_progress=lambda done, total: progress.append((done, total)),
    )

    np.testing.assert_allclose(vectors, FakeEmbedder().embed(texts))
    assert embedder.batch_sizes == [2, 2, 1]
    assert progress == [(2, 5), (4, 5), (5, 5)]
    assert vectors.dtype == np.float32


def test_unnormalized_vectors_are_rejected() -> None:
    with pytest.raises(EmbeddingError, match="normalized"):
        embed_in_batches(FakeEmbedder(normalize=False), ["abc", "de"], batch_size=8)


def test_fingerprint_changes_with_model_ids_or_text() -> None:
    base = fingerprint_embedding_input("m1", ["id1", "id2"], ["good", "bad"])

    assert base == fingerprint_embedding_input("m1", ["id1", "id2"], ["good", "bad"])
    assert base != fingerprint_embedding_input("m2", ["id1", "id2"], ["good", "bad"])
    assert base != fingerprint_embedding_input("m1", ["id1", "id3"], ["good", "bad"])
    assert base != fingerprint_embedding_input("m1", ["id1", "id2"], ["good", "worse"])
