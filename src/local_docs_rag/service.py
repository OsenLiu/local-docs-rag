import json
from collections import defaultdict

from local_docs_rag.models import Answer, SearchResult
from local_docs_rag.config import DEFAULT_PROJECT
from local_docs_rag.providers import AnswerProvider, EmbeddingProvider
from local_docs_rag.store import IndexStore


class SearchService:
    def __init__(self, store: IndexStore, embedding_provider: EmbeddingProvider | None = None) -> None:
        self._store = store
        self._embedding_provider = embedding_provider

    def search(
        self,
        query: str,
        limit: int = 8,
        *,
        project: str = DEFAULT_PROJECT,
    ) -> list[SearchResult]:
        if limit <= 0 or not query.strip():
            return []
        keyword_results = self._store.search_keyword(query, limit, project=project)
        vector_results: list[SearchResult] = []
        if self._embedding_provider is not None and self._store.embedding_dimensions is not None:
            query_vectors = self._embedding_provider.embed([query])
            if query_vectors:
                vector_results = self._store.search_vector(query_vectors[0], limit, project=project)
        return self._reciprocal_rank_fusion([keyword_results, vector_results], limit)

    @staticmethod
    def _reciprocal_rank_fusion(
        result_lists: list[list[SearchResult]],
        limit: int,
        rank_constant: int = 60,
    ) -> list[SearchResult]:
        results_by_id: dict[str, SearchResult] = {}
        scores: dict[str, float] = defaultdict(float)
        for ranked_results in result_lists:
            for rank, result in enumerate(ranked_results, start=1):
                chunk_id = result.chunk.id
                results_by_id.setdefault(chunk_id, result)
                scores[chunk_id] += 1.0 / (rank_constant + rank)
        ranked_ids = sorted(results_by_id, key=lambda chunk_id: scores[chunk_id], reverse=True)
        return [SearchResult(results_by_id[chunk_id].chunk, scores[chunk_id]) for chunk_id in ranked_ids[:limit]]


class RagService:
    _ABSTAIN_MESSAGE = "I could not find supported information in the indexed documents."

    def __init__(self, search_service: SearchService, answer_provider: AnswerProvider) -> None:
        self._search_service = search_service
        self._answer_provider = answer_provider

    def answer(
        self,
        query: str,
        limit: int = 8,
        *,
        project: str = DEFAULT_PROJECT,
    ) -> Answer:
        evidence = self._search_service.search(query, limit, project=project)
        if not evidence:
            return Answer(self._ABSTAIN_MESSAGE, [], True)

        try:
            payload = json.loads(self._answer_provider.answer(query, evidence))
        except (json.JSONDecodeError, TypeError):
            return Answer(self._ABSTAIN_MESSAGE, [], True)

        answer_text = payload.get("answer") if isinstance(payload, dict) else None
        citation_ids = payload.get("citations") if isinstance(payload, dict) else None
        if not isinstance(answer_text, str) or not answer_text.strip() or not isinstance(citation_ids, list):
            return Answer(self._ABSTAIN_MESSAGE, [], True)

        available_ids = {result.chunk.id for result in evidence}
        citations = list(dict.fromkeys(item for item in citation_ids if isinstance(item, str) and item in available_ids))
        if not citations:
            return Answer(self._ABSTAIN_MESSAGE, [], True)
        return Answer(answer_text.strip(), citations, False)