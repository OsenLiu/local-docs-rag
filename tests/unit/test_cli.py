import json
import subprocess
import sys
from pathlib import Path

from local_docs_rag.config import Settings


def run_cli(config_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "local_docs_rag.cli", *arguments, "--config-path", str(config_path)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_index_and_search_commands_share_local_index(tmp_path: Path) -> None:
    source_dir = tmp_path / "documents"
    source_dir.mkdir()
    (source_dir / "guide.md").write_text(
        "# Network Guide\n\nUse a trusted network for setup.\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    Settings(
        source_dir=source_dir,
        database_path=tmp_path / "index.sqlite3",
        base_url=None,
        embedding_model=None,
        embedding_dimensions=None,
        answer_model=None,
    ).save(config_path)

    index_result = run_cli(config_path, "index")
    search_result = run_cli(config_path, "search", "trusted network")

    assert index_result.returncode == 0, index_result.stderr
    assert "added=1" in index_result.stdout
    assert search_result.returncode == 0, search_result.stderr
    assert str(source_dir / "guide.md") in search_result.stdout
    assert "Network Guide" in search_result.stdout


def test_evaluate_command_writes_retrieval_metrics(tmp_path: Path) -> None:
    source_dir = tmp_path / "documents"
    source_dir.mkdir()
    (source_dir / "guide.md").write_text("# Network Guide\n\nUse a trusted network for setup.\n", encoding="utf-8")
    config_path = tmp_path / "config.json"
    Settings(
        source_dir=source_dir,
        database_path=tmp_path / "index.sqlite3",
        base_url=None,
        embedding_model=None,
        embedding_dimensions=None,
        answer_model=None,
    ).save(config_path)
    assert run_cli(config_path, "index").returncode == 0

    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {"query": "trusted network", "expected_source": "guide.md", "expected_locator": "Network Guide", "language": "en"}
        )
        + "\n",
        encoding="utf-8",
    )
    results_path = tmp_path / "results.json"
    evaluate_result = run_cli(
        config_path,
        "evaluate",
        "--questions-path",
        str(questions_path),
        "--output-path",
        str(results_path),
    )

    assert evaluate_result.returncode == 0, evaluate_result.stderr
    metrics = json.loads(results_path.read_text(encoding="utf-8"))
    assert metrics["query_count"] == 1
    assert metrics["recall_at_5"] == 1.0
    assert metrics["locator_accuracy_at_5"] == 1.0