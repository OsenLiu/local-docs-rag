import json
from pathlib import Path

from local_docs_rag.models import DocumentChunk
from local_docs_rag.providers import AnswerProvider, EmbeddingProvider
from local_docs_rag.service import RagService, SearchService
from local_docs_rag.store import IndexStore


def test_search_fuses_keyword_and_vector_results(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "index.sqlite3", embedding_dimensions=2)
    embedding_provider = EmbeddingProvider.create_null(dimensions=2)
    query_vector = embedding_provider.embed(["network setup"])[0]
    orthogonal_vector = [-query_vector[1], query_vector[0]]
    keyword_chunk = DocumentChunk("keyword", "guide.md", "Guide", "network setup steps", "hash")
    vector_chunk = DocumentChunk("vector", "manual.md", "Network", "device connectivity", "hash")
    store.upsert_document("guide.md", "hash", [keyword_chunk], [orthogonal_vector])
    store.upsert_document("manual.md", "hash", [vector_chunk], [query_vector])

    results = SearchService(store, embedding_provider).search("network setup", limit=5)

    assert {result.chunk.id for result in results} == {"keyword", "vector"}


def test_search_uses_keyword_results_without_embedding_provider(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "index.sqlite3")
    chunk = DocumentChunk("keyword", "guide.md", "Guide", "trusted network setup", "hash")
    store.upsert_document("guide.md", "hash", [chunk])

    results = SearchService(store).search("trusted network", limit=5)

    assert [result.chunk.id for result in results] == ["keyword"]


def test_answer_abstains_without_evidence(tmp_path: Path) -> None:
    answer_provider = AnswerProvider.create_null()
    service = RagService(SearchService(IndexStore(tmp_path / "index.sqlite3")), answer_provider)

    answer = service.answer("unknown topic")

    assert answer.abstained is True
    assert answer.citations == []
    assert answer_provider.responses == []


def test_answer_accepts_only_citations_from_retrieved_evidence(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "index.sqlite3")
    chunk = DocumentChunk("evidence-id", "guide.md", "Guide", "Reset takes ten seconds.", "hash")
    store.upsert_document("guide.md", "hash", [chunk])
    answer_provider = AnswerProvider.create_null(
        json.dumps({"answer": "Hold reset for ten seconds.", "citations": ["evidence-id"]})
    )
    service = RagService(SearchService(store), answer_provider)

    answer = service.answer("How long is reset?")

    assert answer.answer == "Hold reset for ten seconds."
    assert answer.citations == ["evidence-id"]
    assert answer.abstained is False


def test_answer_rejects_citations_outside_retrieved_evidence(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "index.sqlite3")
    chunk = DocumentChunk("evidence-id", "guide.md", "Guide", "Reset takes ten seconds.", "hash")
    store.upsert_document("guide.md", "hash", [chunk])
    answer_provider = AnswerProvider.create_null(
        json.dumps({"answer": "The device resets quickly.", "citations": ["invented-id"]})
    )
    service = RagService(SearchService(store), answer_provider)

    answer = service.answer("Reset details")

    assert answer.abstained is True
    assert answer.citations == []