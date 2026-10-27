import hashlib
from collections.abc import Sequence

import tiktoken

from local_docs_rag.models import DocumentChunk


def chunk_sections(
    source: str,
    sections: Sequence[tuple[str, str]],
    document_hash: str,
    project: str = "",
    max_tokens: int = 600,
    overlap_tokens: int = 80,
) -> list[DocumentChunk]:
    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("max_tokens must be positive and greater than overlap_tokens")

    encoding = tiktoken.get_encoding("cl100k_base")
    chunks: list[DocumentChunk] = []
    step = max_tokens - overlap_tokens

    for section_index, (locator, content) in enumerate(sections):
        tokens = encoding.encode(content)
        if not tokens:
            continue

        for chunk_index, start in enumerate(range(0, len(tokens), step)):
            text = encoding.decode(tokens[start : start + max_tokens])
            if not text.strip():
                continue
            chunk_id = hashlib.sha256(
                f"{project}\0{source}\0{locator}\0{section_index}\0{chunk_index}\0{text}".encode("utf-8")
            ).hexdigest()
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    source=source,
                    locator=locator,
                    text=text,
                    document_hash=document_hash,
                )
            )
            if start + max_tokens >= len(tokens):
                break

    return chunks