import sqlite3
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


def test_store_isolates_shared_source_across_projects(tmp_path: Path) -> None:
    store = IndexStore(tmp_path / "index.sqlite3", embedding_dimensions=2)
    alpha_chunk = make_chunk("alpha-chunk", "guide.md", "alpha network instructions")
    beta_chunk = make_chunk("beta-chunk", "guide.md", "beta network instructions")
    store.upsert_document("guide.md", "alpha-hash", [alpha_chunk], [[0.0, 1.0]], project="alpha")
    store.upsert_document("guide.md", "beta-hash", [beta_chunk], [[1.0, 0.0]], project="beta")

    assert [result.chunk.id for result in store.search_keyword("network", 5, project="alpha")] == ["alpha-chunk"]
    assert [result.chunk.id for result in store.search_keyword("network", 5, project="beta")] == ["beta-chunk"]
    assert [result.chunk.id for result in store.search_vector([1.0, 0.0], 1, project="alpha")] == ["alpha-chunk"]
    assert [chunk.id for chunk in store.get_chunks(["beta-chunk", "alpha-chunk"], project="alpha")] == ["alpha-chunk"]
    assert store.document_hashes(project="alpha") == {"guide.md": "alpha-hash"}

    store.remove_document("guide.md", project="alpha")

    assert store.search_keyword("alpha", 5, project="alpha") == []
    assert [result.chunk.id for result in store.search_keyword("beta", 5, project="beta")] == ["beta-chunk"]


def test_store_adopts_legacy_index_under_one_project(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.sqlite3"
    source = str(tmp_path / "guide.md")
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
           CREATE TABLE documents (source TEXT PRIMARY KEY, document_hash TEXT NOT NULL);
           CREATE TABLE chunks (
               vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
               id TEXT NOT NULL UNIQUE,
               source TEXT NOT NULL REFERENCES documents(source) ON DELETE CASCADE,
               locator TEXT NOT NULL,
               text TEXT NOT NULL,
               document_hash TEXT NOT NULL
           );
           CREATE VIRTUAL TABLE chunks_fts USING fts5(
               chunk_id UNINDEXED, source UNINDEXED, locator UNINDEXED, text
           );"""
    )
    connection.execute("INSERT INTO documents(source, document_hash) VALUES (?, ?)", (source, "legacy-hash"))
    connection.execute(
        "INSERT INTO chunks(id, source, locator, text, document_hash) VALUES (?, ?, ?, ?, ?)",
        ("legacy-chunk", source, "Guide", "legacy project material", "legacy-hash"),
    )
    connection.execute(
        "INSERT INTO chunks_fts(chunk_id, source, locator, text) VALUES (?, ?, ?, ?)",
        ("legacy-chunk", source, "Guide", "legacy project material"),
    )
    connection.commit()
    connection.close()

    store = IndexStore(database_path)
    assert store.adopt_legacy_documents("alpha") == 1
    assert store.document_hashes(project="alpha") == {source: "legacy-hash"}
    assert [result.chunk.id for result in store.search_keyword("legacy", 5, project="alpha")] == [
        "legacy-chunk"
    ]