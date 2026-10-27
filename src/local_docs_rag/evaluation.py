from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from local_docs_rag.models import SearchResult


@dataclass(frozen=True)
class RetrievalQuestion:
    query: str
    expected_source: str
    expected_locator: str
    language: str


@dataclass(frozen=True)
class EvaluationMetrics:
    query_count: int
    recall_at_5: float
    locator_accuracy_at_5: float
    chinese_query_count: int
    chinese_recall_at_5: float | None


def _matches_source(source: str, expected_source: str, source_roots: Path | Sequence[Path]) -> bool:
    path = Path(source).resolve()
    expected_path = Path(expected_source)
    if expected_path.is_absolute():
        return path == expected_path.resolve()
    roots = (source_roots,) if isinstance(source_roots, Path) else source_roots
    for source_root in roots:
        try:
            if path.relative_to(source_root.resolve()).as_posix() == expected_source:
                return True
        except ValueError:
            continue
    return path.as_posix() == expected_source


def evaluate_retrieval(
    questions: Iterable[RetrievalQuestion],
    search: Callable[[str, int], list[SearchResult]],
    source_roots: Path | Sequence[Path],
) -> EvaluationMetrics:
    question_list = list(questions)
    source_hits = 0
    locator_hits = 0
    chinese_count = 0
    chinese_hits = 0

    for question in question_list:
        results = search(question.query, 5)
        matching_source = any(
            _matches_source(result.chunk.source, question.expected_source, source_roots)
            for result in results
        )
        matching_locator = any(
            _matches_source(result.chunk.source, question.expected_source, source_roots)
            and result.chunk.locator == question.expected_locator
            for result in results
        )
        source_hits += matching_source
        locator_hits += matching_locator
        if question.language == "zh":
            chinese_count += 1
            chinese_hits += matching_source

    total = len(question_list)
    return EvaluationMetrics(
        query_count=total,
        recall_at_5=source_hits / total if total else 0.0,
        locator_accuracy_at_5=locator_hits / total if total else 0.0,
        chinese_query_count=chinese_count,
        chinese_recall_at_5=chinese_hits / chinese_count if chinese_count else None,
    )