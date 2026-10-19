from pathlib import Path

import numpy as np
import pytest

from rag_poc.embedding.numpy_embedding_store import NumpyEmbeddingStore
from rag_poc.lib import FeedbackEmbeddings


def test_embeddings_round_trip_with_ids_and_fingerprint(tmp_path: Path) -> None:
    vectors = np.eye(3, 4, dtype=np.float32)
    embeddings = FeedbackEmbeddings(
        feedback_ids=("a", "b", "c"), vectors=vectors, model_name="fake-model"
    )
    store = NumpyEmbeddingStore(
        vectors_location=tmp_path / "embeddings.npy",
        manifest_location=tmp_path / "embeddings_manifest.json",
    )
    assert store.stored_fingerprint() is None

    store.save(embeddings, fingerprint="abc123")
    loaded = store.load()

    assert store.stored_fingerprint() == "abc123"
    assert loaded.feedback_ids == ("a", "b", "c")
    assert loaded.model_name == "fake-model"
    np.testing.assert_array_equal(loaded.vectors, vectors)


def test_misaligned_embeddings_are_rejected() -> None:
    with pytest.raises(ValueError, match="2 vectors for 3 feedback ids"):
        FeedbackEmbeddings(
            feedback_ids=("a", "b", "c"),
            vectors=np.zeros((2, 4), dtype=np.float32),
            model_name="fake-model",
        )
