"""Stores the FAISS index, its position -> feedback map and a manifest side by side.

The manifest is written last and read first: a partially written index has no manifest and is
treated as missing, and every load checks that the three files agree with each other.
"""

import json
from pathlib import Path

import pandas as pd

from rag_poc.lib import (
    RetrievalError,
    SearchIndexManifest,
    feedback_items_from_frame,
    feedback_items_to_frame,
)
from rag_poc.retrieval.faiss_vector_index import FaissVectorIndex
from rag_poc.retrieval.retrieval_ports import SearchIndex

_POSITION = "position"


class FaissSearchIndexStore:
    """Implements ``SearchIndexStore`` with ``faiss.index``, ``id_map.parquet`` and a manifest."""

    def __init__(
        self, *, index_location: Path, id_map_location: Path, manifest_location: Path
    ) -> None:
        self._index_location = index_location
        self._id_map_location = id_map_location
        self._manifest_location = manifest_location

    @property
    def location(self) -> Path:
        return self._index_location

    def exists(self) -> bool:
        return all(
            path.is_file()
            for path in (self._index_location, self._id_map_location, self._manifest_location)
        )

    def read_manifest(self) -> SearchIndexManifest | None:
        if not self.exists():
            return None
        data = json.loads(self._manifest_location.read_text(encoding="utf-8"))
        return SearchIndexManifest.from_dict(data)

    def save(self, search_index: SearchIndex) -> None:
        self._manifest_location.unlink(missing_ok=True)  # mark incomplete until fully written
        search_index.vectors.write(self._index_location)
        id_map = feedback_items_to_frame(search_index.feedback)
        id_map.insert(0, _POSITION, range(len(id_map)))
        id_map.to_parquet(self._id_map_location, index=False)
        self._manifest_location.write_text(
            json.dumps(search_index.manifest.to_dict(), indent=2), encoding="utf-8"
        )

    def load(self) -> SearchIndex:
        manifest = self.read_manifest()
        if manifest is None:
            raise RetrievalError(
                f"No search index at {self._index_location}. Run `rag-poc index` first."
            )
        vectors = FaissVectorIndex.read(self._index_location)
        id_map = pd.read_parquet(self._id_map_location).sort_values(_POSITION)
        feedback = feedback_items_from_frame(id_map)

        if not (vectors.size == len(feedback) == manifest.count):
            raise RetrievalError(
                f"Search index files disagree: {vectors.size} vectors, {len(feedback)} mapped "
                f"feedback items, manifest says {manifest.count}. Rebuild with `rag-poc index`."
            )
        if vectors.dimension != manifest.dimension:
            raise RetrievalError(
                f"Index dimension {vectors.dimension} != manifest dimension {manifest.dimension}."
            )
        return SearchIndex(vectors=vectors, feedback=feedback, manifest=manifest)
