import os
from pathlib import Path

from local_docs_rag.models import DocumentChunk
from local_docs_rag.providers import AnswerProvider
from local_docs_rag.service import RagService, SearchService
from local_docs_rag.store import IndexStore
from local_docs_rag.mcp_server import create_server


database_path = Path(os.environ["RAG_TEST_DATABASE_PATH"])
store = IndexStore(database_path)
chunk = DocumentChunk(
    id="reset-chunk",
    source="manual.md",
    locator="Reset > Recovery",
    text="Hold the reset button for ten seconds.",
    document_hash="test-hash",
)
store.upsert_document("manual.md", "test-hash", [chunk])
search_service = SearchService(store)
rag_service = RagService(search_service, AnswerProvider.create_null())
create_server(store, search_service, rag_service).run(transport="stdio")