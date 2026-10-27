import re
import sqlite3
from contextlib import closing
from pathlib import Path
from collections.abc import Sequence

import sqlite_vec

from local_docs_rag.config import DEFAULT_PROJECT
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
            document_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "project" not in document_columns:
                connection.execute("ALTER TABLE documents ADD COLUMN project TEXT NOT NULL DEFAULT ''")
            if "source_path" not in document_columns:
                connection.execute("ALTER TABLE documents ADD COLUMN source_path TEXT")
            connection.execute("UPDATE documents SET source_path = source WHERE source_path IS NULL")

            chunk_columns = {row["name"] for row in connection.execute("PRAGMA table_info(chunks)").fetchall()}
            if "project" not in chunk_columns:
                connection.execute("ALTER TABLE chunks ADD COLUMN project TEXT NOT NULL DEFAULT ''")
            if "source_path" not in chunk_columns:
                connection.execute("ALTER TABLE chunks ADD COLUMN source_path TEXT")
            connection.execute("UPDATE chunks SET source_path = source WHERE source_path IS NULL")
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS documents_project_source_path "
                "ON documents(project, source_path)"
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
        *,
        project: str = DEFAULT_PROJECT,
    ) -> None:
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("embedding count must match chunk count")
        if embeddings is not None and self.embedding_dimensions is None:
            raise ValueError("embedding dimensions must be configured before storing vectors")

        with closing(self._connect()) as connection, connection:
            existing_document = connection.execute(
                "SELECT source FROM documents WHERE project = ? AND source_path = ?",
                (project, source),
            ).fetchone()
            source_key = (
                existing_document["source"]
                if existing_document is not None
                else f"{project}\0{source}"
            )
            self._delete_source(connection, source_key)
            connection.execute(
                """INSERT INTO documents(source, document_hash, project, source_path)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(source) DO UPDATE SET
                       document_hash = excluded.document_hash,
                       project = excluded.project,
                       source_path = excluded.source_path""",
                (source_key, document_hash, project, source),
            )
            for chunk_index, chunk in enumerate(chunks):
                cursor = connection.execute(
                    """INSERT INTO chunks(id, source, locator, text, document_hash, project, source_path)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (chunk.id, source_key, chunk.locator, chunk.text, document_hash, project, source),
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

    def remove_document(self, source: str, *, project: str = DEFAULT_PROJECT) -> None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT source FROM documents WHERE project = ? AND source_path = ?",
                (project, source),
            ).fetchone()
            if row is not None:
                self._delete_source(connection, row["source"])

    def document_hashes(self, *, project: str = DEFAULT_PROJECT) -> dict[str, str]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT source_path, document_hash FROM documents WHERE project = ? ORDER BY source_path",
                (project,),
            ).fetchall()
        return {row["source_path"]: row["document_hash"] for row in rows}

    def remove_project(self, project: str) -> None:
        with closing(self._connect()) as connection, connection:
            sources = connection.execute(
                "SELECT source FROM documents WHERE project = ?",
                (project,),
            ).fetchall()
            for row in sources:
                self._delete_source(connection, row["source"])

    def adopt_legacy_documents(self, project: str) -> int:
        with closing(self._connect()) as connection, connection:
            conflicts = connection.execute(
                """SELECT legacy.source_path
                   FROM documents AS legacy
                   JOIN documents AS existing
                     ON existing.project = ? AND existing.source_path = legacy.source_path
                   WHERE legacy.project = ''""",
                (project,),
            ).fetchone()
            if conflicts is not None:
                raise ValueError(
                    f"cannot adopt legacy index: project {project!r} already contains {conflicts['source_path']}"
                )
            cursor = connection.execute(
                "UPDATE documents SET project = ? WHERE project = ''",
                (project,),
            )
            connection.execute(
                "UPDATE chunks SET project = ? WHERE project = ''",
                (project,),
            )
            return cursor.rowcount

    def get_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        project: str = DEFAULT_PROJECT,
    ) -> list[DocumentChunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""SELECT id, source_path, locator, text, document_hash
                    FROM chunks WHERE id IN ({placeholders}) AND project = ?""",
                (*chunk_ids, project),
            ).fetchall()
        chunks_by_id = {
            row["id"]: DocumentChunk(
                id=row["id"],
                source=row["source_path"],
                locator=row["locator"],
                text=row["text"],
                document_hash=row["document_hash"],
            )
            for row in rows
        }
        return [chunks_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks_by_id]

    def search_keyword(
        self,
        query: str,
        limit: int,
        *,
        project: str = DEFAULT_PROJECT,
    ) -> list[SearchResult]:
        if limit <= 0 or not query.strip():
            return []
        terms = re.findall(r"\w+", query, flags=re.UNICODE)
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term}"' for term in terms)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT c.id, c.source_path, c.locator, c.text, c.document_hash, bm25(chunks_fts) AS rank
                   FROM chunks_fts JOIN chunks AS c ON c.id = chunks_fts.chunk_id
                   WHERE chunks_fts MATCH ? AND c.project = ? ORDER BY rank LIMIT ?""",
                (fts_query, project, limit),
            ).fetchall()
        return [
            SearchResult(
                chunk=DocumentChunk(
                    id=row["id"],
                    source=row["source_path"],
                    locator=row["locator"],
                    text=row["text"],
                    document_hash=row["document_hash"],
                ),
                score=-float(row["rank"]),
            )
            for row in rows
        ]

    def search_vector(
        self,
        vector: Sequence[float],
        limit: int,
        *,
        project: str = DEFAULT_PROJECT,
    ) -> list[SearchResult]:
        if limit <= 0 or self.embedding_dimensions is None:
            return []
        if len(vector) != self.embedding_dimensions:
            raise ValueError("query vector dimension does not match the index")
        with closing(self._connect()) as connection:
            vector_count = connection.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
            if vector_count == 0:
                return []
            rows = connection.execute(
                """SELECT c.id, c.source_path, c.locator, c.text, c.document_hash, v.distance
                   FROM chunk_vectors AS v JOIN chunks AS c ON c.vector_id = v.rowid
                   WHERE v.embedding MATCH ? AND k = ? AND c.project = ?
                   ORDER BY v.distance LIMIT ?""",
                (sqlite_vec.serialize_float32(vector), vector_count, project, limit),
            ).fetchall()
        return [
            SearchResult(
                chunk=DocumentChunk(
                    id=row["id"],
                    source=row["source_path"],
                    locator=row["locator"],
                    text=row["text"],
                    document_hash=row["document_hash"],
                ),
                score=1.0 / (1.0 + float(row["distance"])),
            )
            for row in rows
        ]