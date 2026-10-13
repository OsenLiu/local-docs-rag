from pathlib import Path

from local_docs_rag.evaluation import RetrievalQuestion, evaluate_retrieval
from local_docs_rag.models import DocumentChunk, SearchResult


def test_evaluate_retrieval_reports_source_locator_and_chinese_recall(tmp_path: Path) -> None:
    questions = [
        RetrievalQuestion("firmware setup", "guide.pdf", "page:4", "en"),
        RetrievalQuestion("韌體設定", "other.md", "Setup", "zh"),
    ]
    results = {
        "firmware setup": [
            SearchResult(DocumentChunk("wrong", str(tmp_path / "other.md"), "Intro", "", "hash"), 0.8),
            SearchResult(DocumentChunk("right", str(tmp_path / "guide.pdf"), "page:4", "", "hash"), 0.7),
        ],
        "韌體設定": [
            SearchResult(DocumentChunk("chinese", str(tmp_path / "other.md"), "Setup", "", "hash"), 0.9),
        ],
    }

    metrics = evaluate_retrieval(questions, lambda query, limit: results[query][:limit], tmp_path)

    assert metrics.query_count == 2
    assert metrics.recall_at_5 == 1.0
    assert metrics.locator_accuracy_at_5 == 1.0
    assert metrics.chinese_recall_at_5 == 1.0