from pathlib import Path

from local_docs_rag.models import DocumentChunk
from local_docs_rag.store import IndexStore


def make_chunk(chunk_id: str, source: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        source=source,
        locator="Guide",
        text=text,
        document_hash=f"hash-{source}",
    )


def test_store_supports_keyword_and_vector_search(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "index.sqlite3", embedding_dimensions=2)
    chunk = make_chunk("chunk-1", "guide.md", "trusted network setup")
    store.upsert_document("guide.md", "hash-guide", [chunk], [[1.0, 0.0]])

    keyword_results = store.search_keyword("trusted network", limit=5)
    vector_results = store.search_vector([1.0, 0.0], limit=5)

    assert [result.chunk.id for result in keyword_results] == ["chunk-1"]
    assert [result.chunk.id for result in vector_results] == ["chunk-1"]
    assert keyword_results[0].chunk.locator == "Guide"


def test_upsert_replaces_old_chunks_for_document(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "index.sqlite3", embedding_dimensions=2)
    old_chunk = make_chunk("old-chunk", "guide.md", "legacy setup instructions")
    new_chunk = make_chunk("new-chunk", "guide.md", "current reset instructions")
    store.upsert_document("guide.md", "old-hash", [old_chunk], [[0.0, 1.0]])

    store.upsert_document("guide.md", "new-hash", [new_chunk], [[1.0, 0.0]])

    assert store.search_keyword("legacy", limit=5) == []
    assert [result.chunk.id for result in store.search_keyword("current", limit=5)] == ["new-chunk"]
    assert store.document_hashes() == {"guide.md": "new-hash"}


def test_remove_document_removes_keyword_and_vector_entries(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "index.sqlite3", embedding_dimensions=2)
    chunk = make_chunk("chunk-1", "guide.md", "remove this document")
    store.upsert_document("guide.md", "hash-guide", [chunk], [[1.0, 0.0]])

    store.remove_document("guide.md")

    assert store.search_keyword("remove", limit=5) == []
    assert store.search_vector([1.0, 0.0], limit=5) == []
    assert store.document_hashes() == {}


def test_store_rejects_vector_dimension_mismatch(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "index.sqlite3", embedding_dimensions=2)
    chunk = make_chunk("chunk-1", "guide.md", "vector dimension test")

    try:
        store.upsert_document("guide.md", "hash-guide", [chunk], [[1.0, 0.0, 0.0]])
    except ValueError as error:
        assert "dimension" in str(error)
    else:
        raise AssertionError("expected vector dimension validation")


def test_get_chunks_preserves_requested_order_and_omits_unknown_ids(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "index.sqlite3")
    first = make_chunk("first", "guide.md", "first source text")
    second = make_chunk("second", "guide.md", "second source text")
    store.upsert_document("guide.md", "hash-guide", [first, second])

    chunks = store.get_chunks(["second", "unknown", "first"])

    assert [chunk.id for chunk in chunks] == ["second", "first"]
    assert [chunk.locator for chunk in chunks] == ["Guide", "Guide"]