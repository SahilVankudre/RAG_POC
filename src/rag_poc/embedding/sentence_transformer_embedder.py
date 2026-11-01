"""Sentence Transformers implementation of the ``Embedder`` port.

sentence-transformers (and with it torch, ~10 s to import on CPU) is imported only when the
model is first needed, so commands that reuse stored embeddings never pay for it. The model
is downloaded on first use and cached by Hugging Face.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from rag_poc.lib import EmbeddingError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

KNOWN_QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": _BGE_QUERY_INSTRUCTION,
    "BAAI/bge-base-en-v1.5": _BGE_QUERY_INSTRUCTION,
    "BAAI/bge-large-en-v1.5": _BGE_QUERY_INSTRUCTION,
}
"""Query instructions the model authors recommend; symmetric models such as MiniLM need none."""


class SentenceTransformerEmbedder:
    """Encodes texts with a pretrained Sentence Transformers model, e.g. all-MiniLM-L6-v2."""

    def __init__(
        self, model_name: str, *, query_prefix: str | None = None, device: str | None = None
    ) -> None:
        """``query_prefix=None`` uses the model's known instruction, if any; "" disables it."""
        self._model_name = model_name
        self._query_prefix = (
            KNOWN_QUERY_PREFIXES.get(model_name, "") if query_prefix is None else query_prefix
        )
        self._device = device
        self._model: SentenceTransformer | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def query_prefix(self) -> str:
        return self._query_prefix

    @property
    def dimension(self) -> int:
        dimension = self._loaded_model().get_embedding_dimension()
        if dimension is None:
            raise EmbeddingError(f"{self._model_name} does not report an embedding dimension")
        return int(dimension)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self._loaded_model().encode(
            list(texts),
            batch_size=len(texts),  # callers control batching; encode each call in one pass
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([self._query_prefix + query])[0]

    def _loaded_model(self) -> "SentenceTransformer":
        if self._model is None:
            logger.info("Loading embedding model %s", self._model_name)
            from sentence_transformers import SentenceTransformer
            from transformers.utils import logging as hf_transformers_logging

            hf_transformers_logging.disable_progress_bar()  # clashes with the caller's progress
            try:
                self._model = SentenceTransformer(self._model_name, device=self._device)
            except Exception as error:  # download, auth or incompatible-model errors
                raise EmbeddingError(
                    f"Could not load embedding model {self._model_name}: {error}"
                ) from error
            logger.info(
                "Model ready on %s (dimension %s, max %s tokens)",
                self._model.device,
                self._model.get_embedding_dimension(),
                self._model.max_seq_length,
            )
        return self._model
