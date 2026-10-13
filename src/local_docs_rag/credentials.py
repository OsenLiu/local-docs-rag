import os
from collections.abc import Mapping
from typing import Protocol

import keyring


_SERVICE_NAME = "local-docs-rag"
_ACCOUNT_NAME = "model-provider"


class _Backend(Protocol):
    def get_password(self) -> str | None: ...

    def set_password(self, password: str) -> None: ...


class _KeychainBackend:
    def get_password(self) -> str | None:
        return keyring.get_password(_SERVICE_NAME, _ACCOUNT_NAME)

    def set_password(self, password: str) -> None:
        keyring.set_password(_SERVICE_NAME, _ACCOUNT_NAME, password)


class _MemoryBackend:
    def __init__(self, password: str | None = None) -> None:
        self._password = password

    def get_password(self) -> str | None:
        return self._password

    def set_password(self, password: str) -> None:
        self._password = password


class CredentialStore:
    def __init__(self, backend: _Backend) -> None:
        self._backend = backend

    @classmethod
    def create(cls) -> "CredentialStore":
        return cls(_KeychainBackend())

    @classmethod
    def create_null(cls, initial_api_key: str | None = None) -> "CredentialStore":
        return cls(_MemoryBackend(initial_api_key))

    def get_api_key(self) -> str | None:
        return self._backend.get_password()

    def set_api_key(self, api_key: str) -> None:
        self._backend.set_password(api_key)

    def load_api_key(self, environ: Mapping[str, str] | None = None) -> str | None:
        env = os.environ if environ is None else environ
        return env.get("RAG_API_KEY") or self.get_api_key()