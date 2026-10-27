import json
import subprocess
import sys
from pathlib import Path

from local_docs_rag.config import Settings
from local_docs_rag.store import IndexStore


def run_cli(
    config_path: Path,
    *arguments: str,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "local_docs_rag.cli", *arguments, "--config-path", str(config_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
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
        projects={"manual": []},
    ).save(config_path)

    index_result = run_cli(config_path, "index", "manual", cwd=source_dir)
    search_result = run_cli(config_path, "search", "manual", "trusted network")

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
        projects={"manual": []},
    ).save(config_path)
    assert run_cli(config_path, "index", "manual", cwd=source_dir).returncode == 0
    second_dir = tmp_path / "second-documents"
    second_dir.mkdir()
    (second_dir / "supplement.md").write_text(
        "# Supplement\n\nProject-specific beta material.\n",
        encoding="utf-8",
    )
    assert run_cli(config_path, "index", "manual", cwd=second_dir).returncode == 0

    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        json.dumps(
            {
                "query": "project-specific beta",
                "expected_source": "supplement.md",
                "expected_locator": "Supplement",
                "language": "en",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    results_path = tmp_path / "results.json"
    evaluate_result = run_cli(
        config_path,
        "evaluate",
        "manual",
        "--questions-path",
        str(questions_path),
        "--output-path",
        str(results_path),
    )

    assert evaluate_result.returncode == 0, evaluate_result.stderr
    metrics = json.loads(results_path.read_text(encoding="utf-8"))
    assert metrics["query_count"] == 1
    assert metrics["recall_at_5"] == 1.0


def test_project_commands_register_require_and_remove_project(tmp_path: Path) -> None:
    source_dir = tmp_path / "documents"
    source_dir.mkdir()
    (source_dir / "guide.md").write_text("# Alpha\n\nAlpha-only material.\n", encoding="utf-8")
    config_path = tmp_path / "config.json"
    Settings(
        source_dir=tmp_path / "legacy",
        database_path=tmp_path / "index.sqlite3",
        base_url=None,
        embedding_model=None,
        embedding_dimensions=None,
        answer_model=None,
    ).save(config_path)

    add_result = run_cli(config_path, "project", "add", "alpha")
    assert add_result.returncode == 0, add_result.stderr
    assert run_cli(config_path, "index", "unknown").returncode != 0

    index_result = run_cli(config_path, "index", "alpha", cwd=source_dir)
    assert index_result.returncode == 0, index_result.stderr
    second_dir = tmp_path / "more-documents"
    second_dir.mkdir()
    (second_dir / "extra.md").write_text("# Beta\n\nBeta-only material.\n", encoding="utf-8")
    second_index_result = run_cli(config_path, "index", "alpha", cwd=second_dir)
    assert second_index_result.returncode == 0, second_index_result.stderr
    store = IndexStore(tmp_path / "index.sqlite3")
    assert [result.chunk.source for result in store.search_keyword("alpha", 5, project="alpha")] == [
        str(source_dir / "guide.md")
    ]
    assert [result.chunk.source for result in store.search_keyword("beta", 5, project="alpha")] == [
        str(second_dir / "extra.md")
    ]

    (source_dir / "guide.md").unlink()
    refresh_result = run_cli(config_path, "index", "alpha", cwd=source_dir)
    assert refresh_result.returncode == 0, refresh_result.stderr
    assert store.search_keyword("alpha", 5, project="alpha") == []
    assert store.search_keyword("beta", 5, project="alpha")

    remove_dir_result = run_cli(config_path, "project", "remove-dir", "alpha", str(second_dir), "--yes")
    assert remove_dir_result.returncode == 0, remove_dir_result.stderr
    assert store.search_keyword("beta", 5, project="alpha") == []

    remove_without_confirmation = run_cli(config_path, "project", "remove", "alpha")
    assert remove_without_confirmation.returncode != 0
    assert "--yes" in remove_without_confirmation.stderr

    remove_result = run_cli(config_path, "project", "remove", "alpha", "--yes")
    assert remove_result.returncode == 0, remove_result.stderr
    assert IndexStore(tmp_path / "index.sqlite3").search_keyword("alpha", 5, project="alpha") == []