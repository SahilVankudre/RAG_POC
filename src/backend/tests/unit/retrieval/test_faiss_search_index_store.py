from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from rag_poc.lib import FeedbackItem, RetrievalError, SearchIndexManifest
from rag_poc.retrieval import SearchIndex
from rag_poc.retrieval.faiss_search_index_store import FaissSearchIndexStore
from rag_poc.retrieval.faiss_vector_index import FaissVectorIndex

VECTORS = np.array(
    [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32
)


def _feedback(count: int) -> list[FeedbackItem]:
    return [
        FeedbackItem(feedback_id=f"fb{i}", source="amazon", entity_id=f"B{i}", text=f"review {i}")
        for i in range(count)
    ]


def _manifest(count: int) -> SearchIndexManifest:
    return SearchIndexManifest(
        model_name="fake-model",
        dimension=3,
        count=count,
        index_type="IndexFlatIP",
        embeddings_fingerprint="abc",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _store(tmp_path: Path) -> FaissSearchIndexStore:
    return FaissSearchIndexStore(
        index_location=tmp_path / "faiss.index",
        id_map_location=tmp_path / "id_map.parquet",
        manifest_location=tmp_path / "index_manifest.json",
    )


def test_search_ranks_by_cosine_and_survives_save_and_reload(tmp_path: Path) -> None:
    index = FaissVectorIndex.from_vectors(VECTORS)
    query = np.array([0.8, 0.6, 0.0], dtype=np.float32)

    before = index.search(query, top_k=10)
    store = _store(tmp_path)
    store.save(SearchIndex(vectors=index, feedback=_feedback(4), manifest=_manifest(4)))
    reloaded = store.load()

    assert [position for position, _ in before] == [1, 0, 3, 2]  # top_k clamped to size
    assert before[0][1] == pytest.approx(0.96)
    assert reloaded.vectors.search(query, top_k=10) == before
    assert [item.feedback_id for item in reloaded.feedback] == ["fb0", "fb1", "fb2", "fb3"]
    assert reloaded.manifest == _manifest(4)


def test_inconsistent_index_files_are_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    index = FaissVectorIndex.from_vectors(VECTORS)
    store.save(SearchIndex(vectors=index, feedback=_feedback(3), manifest=_manifest(4)))

    with pytest.raises(RetrievalError, match="disagree"):
        store.load()
