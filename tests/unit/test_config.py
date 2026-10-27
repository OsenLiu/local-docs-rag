import json
from pathlib import Path

import pytest

from local_docs_rag.config import Settings


def test_load_uses_defaults_when_no_config_exists(tmp_path: Path) -> None:
    settings = Settings.load(environ={}, config_path=tmp_path / "missing.json")

    assert settings.source_dir == Path.home() / "Downloads/doc"
    assert settings.database_path == Path.home() / "Library/Application Support/local-docs-rag/index.sqlite3"
    assert settings.embedding_model is None
    assert settings.embedding_dimensions is None
    assert settings.answer_model is None


def test_load_reads_config_and_applies_environment_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "source_dir": "~/Documents/rag-source",
                "database_path": "~/Documents/rag-index.sqlite3",
                "base_url": "https://api.example.test/v1",
                "embedding_model": "embedding-from-file",
                "embedding_dimensions": 1536,
                "answer_model": "answer-from-file",
            }
        ),
        encoding="utf-8",
    )

    settings = Settings.load(
        environ={
            "RAG_SOURCE_DIR": "/tmp/override-source",
            "RAG_EMBEDDING_MODEL": "embedding-from-env",
            "RAG_EMBEDDING_DIMENSIONS": "3072",
        },
        config_path=config_path,
    )

    assert settings.source_dir == Path("/tmp/override-source")
    assert settings.database_path == Path.home() / "Documents/rag-index.sqlite3"
    assert settings.base_url == "https://api.example.test/v1"
    assert settings.embedding_model == "embedding-from-env"
    assert settings.embedding_dimensions == 3072
    assert settings.answer_model == "answer-from-file"


def test_load_rejects_non_positive_embedding_dimensions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        Settings.load(
            environ={"RAG_EMBEDDING_DIMENSIONS": "0"},
            config_path=tmp_path / "missing.json",
        )


def test_load_expands_tilde_in_environment_paths(tmp_path: Path) -> None:
    settings = Settings.load(
        environ={"RAG_SOURCE_DIR": "~/Downloads/doc"},
        config_path=tmp_path / "missing.json",
    )

    assert settings.source_dir == Path.home() / "Downloads/doc"


def test_save_persists_non_secret_settings_and_can_be_loaded(tmp_path: Path) -> None:
    config_path = tmp_path / "settings" / "config.json"
    expected = Settings(
        source_dir=Path("/tmp/documents"),
        database_path=Path("/tmp/rag.sqlite3"),
        base_url="https://api.example.test/v1",
        embedding_model="embedding-model",
        embedding_dimensions=1536,
        answer_model="answer-model",
        projects={"team-docs": Path("/tmp/team-docs")},
    )

    expected.save(config_path)
    actual = Settings.load(environ={}, config_path=config_path)

    assert actual == expected
    assert "api_key" not in config_path.read_text(encoding="utf-8")


def test_load_rejects_invalid_project_names(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"projects": {"Team Docs": str(tmp_path)}}), encoding="utf-8")

    with pytest.raises(ValueError, match="lowercase slug"):
        Settings.load(environ={}, config_path=config_path)