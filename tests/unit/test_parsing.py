from pathlib import Path

from pypdf import PdfWriter

from local_docs_rag.parsing import parse_markdown, parse_pdf


def test_parse_markdown_preserves_heading_hierarchy(tmp_path: Path) -> None:
    markdown_path = tmp_path / "manual.md"
    markdown_path.write_text(
        "# Platform Guide\n\nOverview text.\n\n## Network Setup\n\nConnect the device.\n\n### Verify Connection\n\nCheck the IP address.\n",
        encoding="utf-8",
    )

    sections = parse_markdown(markdown_path)

    assert sections == [
        ("Platform Guide", "Overview text."),
        ("Platform Guide > Network Setup", "Connect the device."),
        ("Platform Guide > Network Setup > Verify Connection", "Check the IP address."),
    ]


def test_parse_markdown_without_headings_uses_document_locator(tmp_path: Path) -> None:
    markdown_path = tmp_path / "notes.md"
    markdown_path.write_text("A standalone note.", encoding="utf-8")

    assert parse_markdown(markdown_path) == [("Document", "A standalone note.")]


def test_parse_pdf_preserves_one_based_page_locations(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as pdf_file:
        writer.write(pdf_file)

    assert parse_pdf(pdf_path) == [("page:1", ""), ("page:2", "")]