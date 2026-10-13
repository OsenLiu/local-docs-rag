import re
import sqlite3
from contextlib import closing
from pathlib import Path
from collections.abc import Sequence

import sqlite_vec

from local_docs_rag.models import DocumentChunk, SearchResult


class IndexStore:
    def __init__(self, database_path: Path, embedding_dimensions: int | None = None) -> None:
        if embedding_dimensions is not None and embedding_dimensions <= 0:
            raise ValueError("embedding dimensions must be a positive integer")
        self.database_path = database_path.expanduser()
        self.embedding_dimensions = embedding_dimensions
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.enable_load_extension(True)
        try:
            sqlite_vec.load(connection)
        finally:
            connection.enable_load_extension(False)
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS documents (source TEXT PRIMARY KEY, document_hash TEXT NOT NULL)"
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS chunks (
                    vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL REFERENCES documents(source) ON DELETE CASCADE,
                    locator TEXT NOT NULL,
                    text TEXT NOT NULL,
                    document_hash TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    source UNINDEXED,
                    locator UNINDEXED,
                    text
                )"""
            )
            existing_dimensions = connection.execute(
                "SELECT value FROM settings WHERE key = 'embedding_dimensions'"
            ).fetchone()
            if existing_dimensions is not None:
                stored_dimension = int(existing_dimensions["value"])
                if self.embedding_dimensions is not None and stored_dimension != self.embedding_dimensions:
                    raise ValueError("embedding dimension does not match the existing index; rebuild the index")
                self.embedding_dimensions = stored_dimension
            elif self.embedding_dimensions is not None:
                connection.execute(
                    "INSERT INTO settings(key, value) VALUES ('embedding_dimensions', ?)",
                    (str(self.embedding_dimensions),),
                )

            if self.embedding_dimensions is not None:
                connection.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(embedding float[{self.embedding_dimensions}])"
                )
            connection.commit()

    def upsert_document(
        self,
        source: str,
        document_hash: str,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> None:
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("embedding count must match chunk count")
        if embeddings is not None and self.embedding_dimensions is None:
            raise ValueError("embedding dimensions must be configured before storing vectors")

        with closing(self._connect()) as connection, connection:
            self._delete_source(connection, source)
            connection.execute(
                "INSERT INTO documents(source, document_hash) VALUES (?, ?) "
                "ON CONFLICT(source) DO UPDATE SET document_hash = excluded.document_hash",
                (source, document_hash),
            )
            for chunk_index, chunk in enumerate(chunks):
                cursor = connection.execute(
                    "INSERT INTO chunks(id, source, locator, text, document_hash) VALUES (?, ?, ?, ?, ?)",
                    (chunk.id, source, chunk.locator, chunk.text, document_hash),
                )
                vector_id = cursor.lastrowid
                connection.execute(
                    "INSERT INTO chunks_fts(chunk_id, source, locator, text) VALUES (?, ?, ?, ?)",
                    (chunk.id, source, chunk.locator, chunk.text),
                )
                if embeddings is not None:
                    vector = embeddings[chunk_index]
                    if len(vector) != self.embedding_dimensions:
                        raise ValueError("embedding vector dimension does not match the index")
                    connection.execute(
                        "INSERT INTO chunk_vectors(rowid, embedding) VALUES (?, ?)",
                        (vector_id, sqlite_vec.serialize_float32(vector)),
                    )

    def _delete_source(self, connection: sqlite3.Connection, source: str) -> None:
        rows = connection.execute("SELECT id, vector_id FROM chunks WHERE source = ?", (source,)).fetchall()
        for row in rows:
            connection.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (row["id"],))
            if self.embedding_dimensions is not None:
                connection.execute("DELETE FROM chunk_vectors WHERE rowid = ?", (row["vector_id"],))
        connection.execute("DELETE FROM chunks WHERE source = ?", (source,))
        connection.execute("DELETE FROM documents WHERE source = ?", (source,))

    def remove_document(self, source: str) -> None:
        with closing(self._connect()) as connection, connection:
            self._delete_source(connection, source)

    def document_hashes(self) -> dict[str, str]:
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT source, document_hash FROM documents ORDER BY source").fetchall()
        return {row["source"]: row["document_hash"] for row in rows}

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[DocumentChunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT id, source, locator, text, document_hash FROM chunks WHERE id IN ({placeholders})",
                tuple(chunk_ids),
            ).fetchall()
        chunks_by_id = {
            row["id"]: DocumentChunk(
                id=row["id"],
                source=row["source"],
                locator=row["locator"],
                text=row["text"],
                document_hash=row["document_hash"],
            )
            for row in rows
        }
        return [chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks_by_id]

    def search_keyword(self, query: str, limit: int) -> list[SearchResult]:
        if limit <= 0 or not query.strip():
            return []
        terms = re.findall(r"\w+", query, flags=re.UNICODE)
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term}"' for term in terms)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT c.id, c.source, c.locator, c.text, c.document_hash, bm25(chunks_fts) AS rank
                   FROM chunks_fts JOIN chunks AS c ON c.id = chunks_fts.chunk_id
                   WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?""",
                (fts_query, limit),
            ).fetchall()
        return [
            SearchResult(
                chunk=DocumentChunk(
                    id=row["id"],
                    source=row["source"],
                    locator=row["locator"],
                    text=row["text"],
                    document_hash=row["document_hash"],
                ),
                score=-float(row["rank"]),
            )
            for row in rows
        ]

    def search_vector(self, vector: Sequence[float], limit: int) -> list[SearchResult]:
        if limit <= 0 or self.embedding_dimensions is None:
            return []
        if len(vector) != self.embedding_dimensions:
            raise ValueError("query vector dimension does not match the index")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT c.id, c.source, c.locator, c.text, c.document_hash, v.distance
                   FROM chunk_vectors AS v JOIN chunks AS c ON c.vector_id = v.rowid
                   WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance""",
                (sqlite_vec.serialize_float32(vector), limit),
            ).fetchall()
        return [
            SearchResult(
                chunk=DocumentChunk(
                    id=row["id"],
                    source=row["source"],
                    locator=row["locator"],
                    text=row["text"],
                    document_hash=row["document_hash"],
                ),
                score=1.0 / (1.0 + float(row["distance"])),
            )
            for row in rows
        ]