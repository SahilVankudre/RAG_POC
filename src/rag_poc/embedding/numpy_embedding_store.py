"""Stores embeddings as a ``.npy`` matrix plus a JSON manifest holding the aligned ids."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rag_poc.lib import FeedbackEmbeddings


class NumpyEmbeddingStore:
    """``embeddings.npy`` holds the vectors; the manifest holds ids, model and fingerprint.

    Implements ``EmbeddingStore``.
    """

    def __init__(self, *, vectors_location: Path, manifest_location: Path) -> None:
        self._vectors_location = vectors_location
        self._manifest_location = manifest_location

    @property
    def location(self) -> Path:
        return self._vectors_location

    def exists(self) -> bool:
        return self._vectors_location.is_file() and self._manifest_location.is_file()

    def stored_fingerprint(self) -> str | None:
        if not self.exists():
            return None
        return self._read_manifest().get("source_fingerprint")

    def save(self, embeddings: FeedbackEmbeddings, *, fingerprint: str) -> None:
        self._vectors_location.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._vectors_location, embeddings.vectors)
        manifest = {
            "model_name": embeddings.model_name,
            "dimension": embeddings.dimension,
            "count": embeddings.count,
            "normalized": True,
            "source_fingerprint": fingerprint,
            "created_at": datetime.now(UTC).isoformat(),
            "feedback_ids": list(embeddings.feedback_ids),
        }
        self._manifest_location.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    def load(self) -> FeedbackEmbeddings:
        manifest = self._read_manifest()
        return FeedbackEmbeddings(
            feedback_ids=tuple(manifest["feedback_ids"]),
            vectors=np.load(self._vectors_location),
            model_name=manifest["model_name"],
        )

    def _read_manifest(self) -> dict[str, Any]:
        return json.loads(self._manifest_location.read_text(encoding="utf-8"))
