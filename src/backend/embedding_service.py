"""
Embedding service for SchemeSathi.
Wraps LangChain's OllamaEmbeddings for both async and sync usage.
"""

import asyncio
from langchain_ollama import OllamaEmbeddings
from backend.config import OLLAMA_HOST, OLLAMA_EMBEDDING_MODEL, EMBEDDING_API_KEY


class EmbeddingService:
    """LangChain OllamaEmbeddings wrapper with sync and async interfaces."""

    def __init__(self):
        # Build optional auth headers for the remote Ollama host
        client_kwargs: dict = {}
        if EMBEDDING_API_KEY:
            client_kwargs["headers"] = {"Authorization": f"Bearer {EMBEDDING_API_KEY}"}

        self._lc_embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            base_url=OLLAMA_HOST,
            client_kwargs=client_kwargs,
        )

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text asynchronously (for query-time use in rag_pipeline)."""
        # OllamaEmbeddings is synchronous; run it in a thread to keep the event loop free.
        return await asyncio.to_thread(self._lc_embeddings.embed_query, text)

    def embed_text_sync(self, text: str) -> list[float]:
        """Embed a single text synchronously (for build_vectordb script)."""
        return self._lc_embeddings.embed_query(text)


embedding_service = EmbeddingService()
