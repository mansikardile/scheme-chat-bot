"""
Embedding service for SchemeSathi.
Wraps LangChain's OllamaEmbeddings for both async and sync usage.
"""

import asyncio
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None

from backend.config import OLLAMA_HOST, OLLAMA_EMBEDDING_MODEL, EMBEDDING_API_KEY


class EmbeddingService:
    """LangChain OllamaEmbeddings wrapper with sync and async interfaces."""

    def __init__(self):
        self.model = OLLAMA_EMBEDDING_MODEL
        self.base_url = OLLAMA_HOST
        self.api_key = EMBEDDING_API_KEY
        self.is_local = True
        self._init_embeddings()

    def _init_embeddings(self):
        if OllamaEmbeddings is None:
            self._lc_embeddings = None
            return

        client_kwargs: dict = {}
        if self.api_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {self.api_key}"}

        try:
            self._lc_embeddings = OllamaEmbeddings(
                model=self.model,
                base_url=self.base_url,
                client_kwargs=client_kwargs if client_kwargs else None,
            )
        except Exception as e:
            print(f"Notice: OllamaEmbeddings init deferred ({e}).")
            self._lc_embeddings = None

    def get_config(self) -> dict:
        """Get current embedding service configuration."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "is_local": self.is_local,
        }

    def update_config(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        is_local: bool | None = None,
    ):
        """Update active embedding configuration at runtime."""
        if model is not None:
            self.model = model
        if is_local is not None:
            self.is_local = is_local
        if base_url is not None:
            self.base_url = base_url
        if api_key is not None:
            self.api_key = api_key

        self._init_embeddings()
        print(f"[EmbeddingService] Config updated -> model: '{self.model}', base_url: '{self.base_url}', is_local: {self.is_local}")

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text asynchronously (for query-time use in rag_pipeline)."""
        # OllamaEmbeddings is synchronous; run it in a thread to keep the event loop free.
        return await asyncio.to_thread(self._lc_embeddings.embed_query, text)

    def embed_text_sync(self, text: str) -> list[float]:
        """Embed a single text synchronously (for build_vectordb script)."""
        return self._lc_embeddings.embed_query(text)


embedding_service = EmbeddingService()

