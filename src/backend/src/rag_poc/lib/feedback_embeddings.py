"""Embedding vectors aligned with the feedback they encode, shared by embedding and retrieval."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class FeedbackEmbeddings:
    """Row ``i`` of ``vectors`` is the embedding of the feedback with id ``feedback_ids[i]``.

    Vectors are float32 and L2-normalized, so inner product equals cosine similarity.
    """

    feedback_ids: tuple[str, ...]
    vectors: np.ndarray
    model_name: str

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {self.vectors.shape}")
        if self.vectors.shape[0] != len(self.feedback_ids):
            raise ValueError(
                f"{self.vectors.shape[0]} vectors for {len(self.feedback_ids)} feedback ids"
            )
        if self.vectors.dtype != np.float32:
            raise ValueError(f"vectors must be float32, got {self.vectors.dtype}")
        if not np.isfinite(self.vectors).all():
            raise ValueError("vectors contain NaN or infinite values")
        if len(set(self.feedback_ids)) != len(self.feedback_ids):
            raise ValueError("feedback ids must be unique")

    @property
    def count(self) -> int:
        return len(self.feedback_ids)

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])


@dataclass(frozen=True, slots=True)
class EmbeddingRunReport:
    """Summary of one embedding run, or of reusing embeddings that were already up to date."""

    model_name: str
    count: int
    dimension: int
    batch_size: int
    seconds: float
    reused: bool
    location: Path
