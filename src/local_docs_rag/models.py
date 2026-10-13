from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    source: str
    locator: str
    text: str
    document_hash: str


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: list[str]
    abstained: bool