from local_docs_rag.providers import EmbeddingProvider


def test_null_embedding_provider_is_deterministic_and_tracks_inputs() -> None:
    provider = EmbeddingProvider.create_null(dimensions=4)

    vectors = provider.embed(["trusted network", "trusted network", "device reset"])

    assert vectors[0] == vectors[1]
    assert vectors[0] != vectors[2]
    assert all(len(vector) == 4 for vector in vectors)
    assert provider.embedded_texts == ["trusted network", "trusted network", "device reset"]