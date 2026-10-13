import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from local_docs_rag.chunking import chunk_sections
from local_docs_rag.parsing import parse_markdown, parse_pdf
from local_docs_rag.providers import EmbeddingProvider
from local_docs_rag.store import IndexStore


@dataclass
class IndexReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: int = 0
    issues: list[str] = field(default_factory=list)


class Indexer:
    def __init__(
        self,
        source_dir: Path,
        store: IndexStore,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.source_dir = source_dir.expanduser().resolve()
        self.store = store
        self.embedding_provider = embedding_provider

    def run(self) -> IndexReport:
        if not self.source_dir.is_dir():
            raise FileNotFoundError(f"document directory does not exist: {self.source_dir}")

        report = IndexReport()
        existing_hashes = self.store.document_hashes()
        files = sorted(
            path
            for path in self.source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".md"}
        )
        discovered_sources = {str(path.resolve()) for path in files}

        for source in existing_hashes.keys() - discovered_sources:
            self.store.remove_document(source)
            report.removed += 1

        for path in files:
            source = str(path.resolve())
            document_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if existing_hashes.get(source) == document_hash:
                report.unchanged += 1
                continue

            try:
                sections = parse_pdf(path) if path.suffix.lower() == ".pdf" else parse_markdown(path)
                chunks = chunk_sections(source, sections, document_hash)
                if not chunks:
                    raise ValueError("no text could be extracted from document")
                embeddings = None
                if self.embedding_provider is not None:
                    embeddings = self.embedding_provider.embed([chunk.text for chunk in chunks])
                self.store.upsert_document(source, document_hash, chunks, embeddings)
            except Exception as error:
                report.failed += 1
                report.issues.append(f"{source}: {error}")
                continue

            if source in existing_hashes:
                report.updated += 1
            else:
                report.added += 1

        return report