import tiktoken

from local_docs_rag.chunking import chunk_sections


def test_chunk_sections_skips_empty_content() -> None:
    chunks = chunk_sections("guide.md", [("Guide", "  \n")], "document-hash")

    assert chunks == []


def test_chunk_sections_respects_token_limit_and_overlap() -> None:
    encoding = tiktoken.get_encoding("cl100k_base")
    text = " ".join(f"term{index}" for index in range(1500))

    chunks = chunk_sections(
        "guide.md",
        [("Guide", text)],
        "document-hash",
        max_tokens=600,
        overlap_tokens=80,
    )

    token_groups = [encoding.encode(chunk.text) for chunk in chunks]
    assert len(chunks) > 1
    assert all(len(tokens) <= 600 for tokens in token_groups)
    assert all(chunk.source == "guide.md" and chunk.locator == "Guide" for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert token_groups[0][-80:] == token_groups[1][:80]