"""Typed application settings, overridable via ``RAG_*`` environment variables or a ``.env`` file.

Example: ``RAG_SAMPLE_SIZE=50000 uv run rag-poc ...`` scales the sample without code changes.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, SettingsError

from rag_poc.lib import ConfigurationError

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class AppSettings(BaseSettings):
    """Every tunable of the pipeline in one validated object, built once by the composition root."""

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Storage — all runtime artifacts live under data_dir (gitignored).
    data_dir: Path = Path("data")

    # Dataset acquisition and sampling (spec §2–3).
    hf_dataset_repo: str = "McAuley-Lab/Amazon-Reviews-2023"
    amazon_category: str = "All_Beauty"
    hf_cache_dir: Path | None = None  # None: the standard Hugging Face cache (~/.cache/huggingface)
    sample_size: int = Field(default=10_000, gt=0)
    random_seed: int = 42

    # Embedding generation (spec §7).
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=64, gt=0)
    # Prepended to queries only. None: the model's recommended instruction (BGE) or none
    # (MiniLM); "" forces no prefix.
    embedding_query_prefix: str | None = None

    # Semantic search (spec §8).
    default_top_k: int = Field(default=5, gt=0)
    # Reviews shorter than this many words ("regular customer") are skipped by default.
    search_min_words: int = Field(default=3, ge=0)

    # LLM: any API that speaks the OpenAI chat completions format. The README lists presets.
    llm_base_url: str = "http://localhost:11434/v1"  # local Ollama
    llm_api_key: SecretStr | None = None  # local servers don't need one
    llm_model_name: str | None = None  # no default: which models exist depends on the provider
    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    llm_max_output_tokens: int = Field(default=1000, gt=0)
    llm_timeout_seconds: float = Field(default=120, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    # Provider-specific request fields, e.g. {"thinking": {"type": "disabled"}} for DeepSeek.
    llm_extra_body: dict[str, Any] = Field(default_factory=dict)

    # RAG answers.
    answer_top_k: int = Field(default=10, gt=0)
    # Limits each review in the context, so one long review can't crowd out the others.
    answer_review_max_chars: int = Field(default=800, gt=0)
    # Below this mean similarity of the top 10 reviews overall, questions are refused without
    # calling the LLM. On-topic questions measured 0.378+ and off-topic ones 0.278 or less.
    answer_min_relevance: float = Field(default=0.33, ge=0, le=1)

    # Evaluation: the question set is versioned, while run results are runtime artifacts.
    evaluation_questions_path: Path = Path("evaluation/test_questions.json")

    log_level: LogLevel = "INFO"

    @field_validator("data_dir")
    @classmethod
    def _resolve_data_dir(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase_log_level(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value

    @property
    def raw_data_dir(self) -> Path:
        """Sampled source data in its native columns."""
        return self.data_dir / "raw"

    @property
    def review_sample_path(self) -> Path:
        """Sample file named by category, size and seed, e.g. ``all_beauty_10k_seed42.parquet``.

        Size and seed are part of the name so changing either never silently reuses a stale
        sample.
        """
        size = self.sample_size
        size_label = f"{size // 1000}k" if size % 1000 == 0 else str(size)
        file_name = f"{self.amazon_category.lower()}_{size_label}_seed{self.random_seed}.parquet"
        return self.raw_data_dir / file_name

    @property
    def processed_data_dir(self) -> Path:
        """Cleaned ``FeedbackItem`` records and the data-quality report."""
        return self.data_dir / "processed"

    @property
    def prepared_feedback_path(self) -> Path:
        return self.processed_data_dir / "feedback_items.parquet"

    @property
    def quality_report_path(self) -> Path:
        return self.processed_data_dir / "quality_report.json"

    @property
    def index_dir(self) -> Path:
        """FAISS index, id mapping, embeddings and manifest."""
        return self.data_dir / "index"

    @property
    def embeddings_path(self) -> Path:
        return self.index_dir / "embeddings.npy"

    @property
    def embeddings_manifest_path(self) -> Path:
        return self.index_dir / "embeddings_manifest.json"

    @property
    def faiss_index_path(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def index_id_map_path(self) -> Path:
        """Vector position -> ``FeedbackItem`` mapping."""
        return self.index_dir / "id_map.parquet"

    @property
    def index_manifest_path(self) -> Path:
        return self.index_dir / "index_manifest.json"

    @property
    def evaluation_runs_dir(self) -> Path:
        """Full results of each evaluation run, for manual review."""
        return self.data_dir / "evaluation"


def load_settings(**overrides: Any) -> AppSettings:
    """Build settings from defaults < ``.env`` < environment < explicit overrides."""
    try:
        return AppSettings(**overrides)
    except (ValidationError, SettingsError) as error:  # SettingsError: malformed JSON values
        raise ConfigurationError(f"Invalid configuration:\n{error}") from error
