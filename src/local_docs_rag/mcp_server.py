import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from local_docs_rag.config import DEFAULT_CONFIG_PATH, Settings
from local_docs_rag.models import SearchResult
from local_docs_rag.providers import AnswerProvider, EmbeddingProvider
from local_docs_rag.service import RagService, SearchService
from local_docs_rag.store import IndexStore


def create_server(store: IndexStore, search_service: SearchService, rag_service: RagService) -> FastMCP:
    server = FastMCP("local-docs-rag")

    @server.tool()
    def search_documents(query: str, limit: int = 8) -> dict[str, object]:
        results: list[SearchResult] = search_service.search(query, limit)
        return {
            "results": [
                {
                    "chunk_id": result.chunk.id,
                    "source": result.chunk.source,
                    "locator": result.chunk.locator,
                    "text": result.chunk.text,
                    "score": result.score,
                }
                for result in results
            ]
        }

    @server.tool()
    def answer_documents(query: str, limit: int = 8) -> dict[str, object]:
        answer = rag_service.answer(query, limit)
        chunks = store.get_chunks(answer.citations)
        return {
            "answer": answer.answer,
            "citations": [
                {"chunk_id": chunk.id, "source": chunk.source, "locator": chunk.locator}
                for chunk in chunks
            ],
            "abstained": answer.abstained,
        }

    return server


def serve(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    settings = Settings.load(config_path=config_path)
    store = IndexStore(settings.database_path, settings.embedding_dimensions)
    if settings.embedding_model is None and settings.embedding_dimensions is None:
        embedding_provider = None
    else:
        embedding_provider = EmbeddingProvider.create(settings)
    if settings.answer_model is None:
        answer_provider = AnswerProvider.create_null(json.dumps({"answer": "", "citations": []}))
    else:
        answer_provider = AnswerProvider.create(settings)
    search_service = SearchService(store, embedding_provider)
    rag_service = RagService(search_service, answer_provider)
    create_server(store, search_service, rag_service).run(transport="stdio")


if __name__ == "__main__":
    serve()