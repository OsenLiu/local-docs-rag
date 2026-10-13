from pathlib import Path

from pypdf import PdfWriter

from local_docs_rag.indexing import Indexer
from local_docs_rag.providers import EmbeddingProvider
from local_docs_rag.store import IndexStore


def test_indexer_indexes_markdown_and_ignores_other_file_types(tmp_path: Path) -> None:
    source_dir = tmp_path / "documents"
    source_dir.mkdir()
    (source_dir / "guide.md").write_text(
        "# Network Guide\n\nUse a trusted network for setup.\n",
        encoding="utf-8",
    )
    (source_dir / "ignored.patch").write_text("not indexed", encoding="utf-8")
    store = IndexStore(tmp_path / "index.sqlite3", embedding_dimensions=4)
    provider = EmbeddingProvider.create_null(dimensions=4)

    report = Indexer(source_dir, store, provider).run()

    assert report.added == 1
    assert report.failed == 0
    assert len(store.search_keyword("trusted network", limit=5)) == 1
    assert len(provider.embedded_texts) == 1


def test_indexer_skips_unchanged_files_and_replaces_modified_content(tmp_path: Path) -> None:
    source_dir = tmp_path / "documents"
    source_dir.mkdir()
    document = source_dir / "guide.md"
    document.write_text("# Setup\n\nLegacy network setup.\n", encoding="utf-8")
    store = IndexStore(tmp_path / "index.sqlite3", embedding_dimensions=4)
    provider = EmbeddingProvider.create_null(dimensions=4)
    indexer = Indexer(source_dir, store, provider)

    first_report = indexer.run()
    embedded_count = len(provider.embedded_texts)
    unchanged_report = indexer.run()
    document.write_text("# Setup\n\nCurrent reset procedure.\n", encoding="utf-8")
    updated_report = indexer.run()

    assert first_report.added == 1
    assert unchanged_report.unchanged == 1
    assert len(provider.embedded_texts) == embedded_count + 1
    assert updated_report.updated == 1
    assert store.search_keyword("legacy", limit=5) == []
    assert len(store.search_keyword("current reset", limit=5)) == 1


def test_indexer_removes_deleted_sources_and_reports_empty_pdf(tmp_path: Path) -> None:
    source_dir = tmp_path / "documents"
    source_dir.mkdir()
    markdown_path = source_dir / "guide.md"
    markdown_path.write_text("# Guide\n\nRemove this document.\n", encoding="utf-8")
    pdf_path = source_dir / "empty.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as pdf_file:
        writer.write(pdf_file)

    store = IndexStore(tmp_path / "index.sqlite3", embedding_dimensions=4)
    provider = EmbeddingProvider.create_null(dimensions=4)
    report = Indexer(source_dir, store, provider).run()
    markdown_path.unlink()
    next_report = Indexer(source_dir, store, provider).run()

    assert report.added == 1
    assert report.failed == 1
    assert any("empty.pdf" in issue for issue in report.issues)
    assert next_report.removed == 1
    assert store.search_keyword("remove", limit=5) == []