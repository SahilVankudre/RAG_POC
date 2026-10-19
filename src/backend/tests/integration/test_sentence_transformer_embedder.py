"""Loads the real embedding model (downloaded on first run). Run with: pytest -m integration."""

import numpy as np
import pytest

from rag_poc.embedding import embed_in_batches

pytestmark = pytest.mark.integration


def test_similar_complaints_embed_closer_than_unrelated_feedback() -> None:
    # Imported here so collecting the default (unit) test run never pays torch's import cost.
    from rag_poc.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    texts = [
        "The battery stopped charging after two months.",
        "The product has poor battery life.",
        "Delivery was fast and packaging was excellent.",
    ]

    vectors = embed_in_batches(embedder, texts, batch_size=2)

    assert vectors.shape == (3, 384)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    battery_vs_battery = float(vectors[0] @ vectors[1])
    battery_vs_delivery = float(vectors[0] @ vectors[2])
    assert battery_vs_battery > battery_vs_delivery + 0.2
