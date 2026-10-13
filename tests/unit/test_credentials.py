from local_docs_rag.credentials import CredentialStore


def test_set_api_key_persists_value_in_store() -> None:
    credentials = CredentialStore.create_null()

    credentials.set_api_key("test-key")

    assert credentials.get_api_key() == "test-key"


def test_environment_api_key_takes_precedence_over_keychain_value() -> None:
    credentials = CredentialStore.create_null("keychain-key")

    assert credentials.load_api_key({"RAG_API_KEY": "environment-key"}) == "environment-key"


def test_missing_environment_api_key_falls_back_to_keychain() -> None:
    credentials = CredentialStore.create_null("keychain-key")

    assert credentials.load_api_key({}) == "keychain-key"