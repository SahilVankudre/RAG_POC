from pathlib import Path

import pytest

from rag_poc.bootstrap import load_settings
from rag_poc.lib import ConfigurationError


def test_artifact_directories_are_derived_from_data_dir(tmp_path: Path) -> None:
    settings = load_settings(_env_file=None, data_dir=tmp_path)

    assert settings.raw_data_dir == tmp_path / "raw"
    assert settings.processed_data_dir == tmp_path / "processed"
    assert settings.index_dir == tmp_path / "index"


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_SAMPLE_SIZE", "50000")
    monkeypatch.setenv("RAG_LOG_LEVEL", "debug")

    settings = load_settings(_env_file=None)

    assert settings.sample_size == 50_000
    assert settings.log_level == "DEBUG"


def test_invalid_values_raise_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="sample_size"):
        load_settings(_env_file=None, sample_size=0)
