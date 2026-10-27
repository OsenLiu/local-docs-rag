import json
from pathlib import Path
from collections.abc import Mapping

from mcp.server.fastmcp import FastMCP

from local_docs_rag.config import DEFAULT_CONFIG_PATH, Settings, validate_project_name
from local_docs_rag.models import SearchResult
from local_docs_rag.providers import AnswerProvider, EmbeddingProvider
from local_docs_rag.service import RagService, SearchService
from local_docs_rag.store import IndexStore


def create_server(
    store: IndexStore,
    search_service: SearchService,
    rag_service: RagService,
    projects: Mapping[str, Path],
) -> FastMCP:
    server = FastMCP("local-docs-rag")

    def require_project(project: str) -> None:
        validate_project_name(project)
        if project not in projects:
            raise ValueError(f"unknown project {project!r}")

    @server.tool()
    def search_documents(project: str, query: str, limit: int = 8) -> dict[str, object]:
        require_project(project)
        results: list[SearchResult] = search_service.search(query, limit, project=project)
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
    def answer_documents(project: str, query: str, limit: int = 8) -> dict[str, object]:
        require_project(project)
        answer = rag_service.answer(query, limit, project=project)
        chunks = store.get_chunks(answer.citations, project=project)
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
    create_server(store, search_service, rag_service, settings.projects).run(transport="stdio")


if __name__ == "__main__":
    serve()