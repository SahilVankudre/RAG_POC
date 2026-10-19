"""Index building and search use cases, wired with in-memory stores and a fake embedder."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from rag_poc.application.feedback_search import FeedbackSearch
from rag_poc.application.search_index_building import SearchIndexBuilding
from rag_poc.embedding import fingerprint_embedding_input
from rag_poc.lib import (
    FeedbackEmbeddings,
    FeedbackItem,
    PreparedFeedback,
    RetrievalError,
    SearchFilters,
    SearchIndexManifest,
)
from rag_poc.retrieval import SearchIndex
from rag_poc.retrieval.faiss_vector_index import FaissVectorIndex

MODEL = "fake-model"
# Unit vectors per text; queries reuse them so expected neighbours are unambiguous.
TEXT_VECTORS = {
    "battery died in a week": [1.0, 0.0, 0.0],
    "charger stopped working": [0.8, 0.6, 0.0],
    "lovely floral scent": [0.0, 0.0, 1.0],
}


class FakeEmbedder:
    model_name = MODEL
    dimension = 3

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([TEXT_VECTORS[text] for text in texts], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


class InMemoryProcessedStore:
    location = Path("prepared.parquet")

    def __init__(self, prepared: list[PreparedFeedback]) -> None:
        self.prepared = prepared

    def exists(self) -> bool:
        return True

    def load(self) -> list[PreparedFeedback]:
        return self.prepared


class InMemoryEmbeddingStore:
    location = Path("embeddings.npy")

    def __init__(self, embeddings: FeedbackEmbeddings, fingerprint: str) -> None:
        self.embeddings = embeddings
        self.fingerprint = fingerprint

    def stored_fingerprint(self) -> str | None:
        return self.fingerprint

    def load(self) -> FeedbackEmbeddings:
        return self.embeddings


class InMemoryIndexStore:
    location = Path("faiss.index")

    def __init__(self) -> None:
        self.saved: SearchIndex | None = None
        self.saves = 0

    def read_manifest(self) -> SearchIndexManifest | None:
        return self.saved.manifest if self.saved else None

    def save(self, search_index: SearchIndex) -> None:
        self.saved = search_index
        self.saves += 1

    def load(self) -> SearchIndex:
        assert self.saved is not None
        return self.saved


def _prepared() -> list[PreparedFeedback]:
    return [
        PreparedFeedback(
            item=FeedbackItem(
                feedback_id=f"fb{i}", source="amazon", entity_id=f"B{i}", text=text, rating=i + 1.0
            ),
            embedding_text=text,
        )
        for i, text in enumerate(TEXT_VECTORS)
    ]


def _wire(prepared: list[PreparedFeedback]) -> tuple[SearchIndexBuilding, InMemoryIndexStore]:
    """Embeddings always come from the original feedback; ``prepared`` is what's on disk now."""
    original = _prepared()
    order = [2, 0, 1]  # embeddings deliberately stored in a different order
    embeddings = FeedbackEmbeddings(
        feedback_ids=tuple(original[i].feedback_id for i in order),
        vectors=FakeEmbedder().embed([original[i].embedding_text for i in order]),
        model_name=MODEL,
    )
    fingerprint = fingerprint_embedding_input(
        MODEL, [p.feedback_id for p in original], [p.embedding_text for p in original]
    )
    index_store = InMemoryIndexStore()
    building = SearchIndexBuilding(
        embedding_store=InMemoryEmbeddingStore(embeddings, fingerprint),
        processed_store=InMemoryProcessedStore(prepared),
        index_store=index_store,
        index_factory=FaissVectorIndex.from_vectors,
    )
    return building, index_store


def test_index_maps_positions_to_the_right_feedback_and_search_ranks_it() -> None:
    building, index_store = _wire(_prepared())
    building.build_index()
    assert building.build_index().reused is True and index_store.saves == 1

    search = FeedbackSearch(index_store=index_store, embedder=FakeEmbedder(), default_top_k=2)
    results = search.search("  battery died in a week ")

    assert [(r.rank, r.feedback_id, r.text) for r in results] == [
        (1, "fb0", "battery died in a week"),
        (2, "fb1", "charger stopped working"),
    ]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.8)
    assert results[0].rating == 1.0 and results[0].entity_id == "B0"


def test_stale_embeddings_are_not_indexed() -> None:
    changed = _prepared()
    changed[0] = PreparedFeedback(item=changed[0].item, embedding_text="edited text")
    building, _ = _wire(changed)

    with pytest.raises(RetrievalError, match="different prepared feedback"):
        building.build_index()


def test_search_rejects_empty_queries_and_mismatched_models() -> None:
    building, index_store = _wire(_prepared())
    building.build_index()

    class OtherModelEmbedder(FakeEmbedder):
        model_name = "another-model"

    with pytest.raises(RetrievalError, match="empty"):
        FeedbackSearch(index_store=index_store, embedder=FakeEmbedder(), default_top_k=5).search(
            "   "
        )
    with pytest.raises(RetrievalError, match="built with fake-model"):
        FeedbackSearch(
            index_store=index_store, embedder=OtherModelEmbedder(), default_top_k=5
        ).search("battery died in a week")


def test_filters_restrict_results_by_rating_and_length() -> None:
    building, index_store = _wire(_prepared())  # ratings: fb0=1, fb1=2, fb2=3
    building.build_index()
    search = FeedbackSearch(index_store=index_store, embedder=FakeEmbedder(), default_top_k=5)

    complaints = search.search("battery died in a week", filters=SearchFilters(min_rating=2))
    wordy = search.search("battery died in a week", filters=SearchFilters(min_words=4))

    assert [r.feedback_id for r in complaints] == ["fb1", "fb2"]
    assert [r.feedback_id for r in wordy] == ["fb0"]


def test_selective_filters_widen_the_candidate_pool_until_enough_results_pass() -> None:
    size = 500
    vectors = np.zeros((size, 2), dtype=np.float32)
    vectors[:, 0] = np.linspace(1.0, 0.0, size)  # position 0 is most similar to the query
    vectors[:, 1] = np.sqrt(1.0 - vectors[:, 0] ** 2)
    feedback = [
        FeedbackItem(
            feedback_id=f"fb{i}",
            source="test",
            entity_id="E",
            text="only the last two are complaints",
            rating=1.0 if i >= size - 2 else 5.0,
        )
        for i in range(size)
    ]
    index = SearchIndex(
        vectors=FaissVectorIndex.from_vectors(vectors),
        feedback=feedback,
        manifest=SearchIndexManifest(
            model_name=MODEL,
            dimension=2,
            count=size,
            index_type="IndexFlatIP",
            embeddings_fingerprint="x",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )
    store = InMemoryIndexStore()
    store.saved = index

    class QueryEmbedder(FakeEmbedder):
        def embed_query(self, query: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    results = FeedbackSearch(index_store=store, embedder=QueryEmbedder(), default_top_k=2).search(
        "anything", filters=SearchFilters(max_rating=2)
    )

    assert [r.feedback_id for r in results] == [f"fb{size - 2}", f"fb{size - 1}"]
