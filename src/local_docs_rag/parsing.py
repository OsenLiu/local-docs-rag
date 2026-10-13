from pathlib import Path

from markdown_it import MarkdownIt
from pypdf import PdfReader


def parse_pdf(path: Path) -> list[tuple[str, str]]:
    reader = PdfReader(path)
    return [(f"page:{page_number}", page.extract_text() or "") for page_number, page in enumerate(reader.pages, 1)]


def parse_markdown(path: Path) -> list[tuple[str, str]]:
    tokens = MarkdownIt().parse(path.read_text(encoding="utf-8"))
    heading_stack: list[str] = []
    current_locator: str | None = None
    current_parts: list[str] = []
    sections: list[tuple[str, str]] = []
    heading_inline_index: int | None = None

    def append_section() -> None:
        if current_locator is not None:
            content = "\n".join(part.strip() for part in current_parts if part.strip()).strip()
            if content:
                sections.append((current_locator, content))

    for token_index, token in enumerate(tokens):
        if token.type == "heading_open":
            append_section()
            level = int(token.tag[1:])
            heading_stack[:] = heading_stack[: level - 1]
            title = tokens[token_index + 1].content.strip()
            heading_stack.append(title)
            current_locator = " > ".join(heading_stack)
            current_parts = []
            heading_inline_index = token_index + 1
        elif token.type == "inline" and token_index != heading_inline_index:
            current_locator = current_locator or "Document"
            current_parts.append(token.content)
        elif token.type in {"fence", "code_block"}:
            current_locator = current_locator or "Document"
            current_parts.append(token.content)

    append_section()
    return sections