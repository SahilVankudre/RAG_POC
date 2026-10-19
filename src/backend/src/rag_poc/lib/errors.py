"""Application-wide exception hierarchy."""


class RagPocError(Exception):
    """Base class for all expected, user-reportable errors raised by rag_poc."""


class ConfigurationError(RagPocError):
    """Raised when settings are missing, invalid or inconsistent."""


class DatasetError(RagPocError):
    """Raised when source data cannot be fetched, read or sampled as requested."""


class EmbeddingError(RagPocError):
    """Raised when the embedding model cannot be loaded or produces unusable vectors."""


class RetrievalError(RagPocError):
    """Raised when the search index is missing, inconsistent or built with a different model."""
