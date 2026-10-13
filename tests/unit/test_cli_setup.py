import json
import os
import subprocess
import sys
from pathlib import Path


def test_setup_writes_shared_non_secret_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    answers = "\n\n\nmultilingual-embedding\n1536\nanswer-model\n"
    home = tmp_path / "home"
    environment = os.environ | {"HOME": str(home), "RAG_API_KEY": "must-not-be-saved"}

    result = subprocess.run(
        [sys.executable, "-m", "local_docs_rag.cli", "setup", "--config-path", str(config_path)],
        input=answers,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    assert settings["source_dir"] == str(home / "Downloads/doc")
    assert settings["embedding_model"] == "multilingual-embedding"
    assert settings["embedding_dimensions"] == 1536
    assert settings["answer_model"] == "answer-model"
    assert "must-not-be-saved" not in config_path.read_text(encoding="utf-8")


def test_setup_defaults_writes_local_paths_without_prompting(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    environment = os.environ | {"HOME": str(tmp_path / "home")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "local_docs_rag.cli",
            "setup",
            "--defaults",
            "--config-path",
            str(config_path),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    assert settings["source_dir"] == str(tmp_path / "home" / "Downloads/doc")
    assert settings["embedding_model"] is None