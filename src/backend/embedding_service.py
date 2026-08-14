"""
Embedding service for SchemeSathi.
Wraps the local Ollama embedding API for both async and sync usage.
"""

from ollama import Client, AsyncClient
from backend.config import OLLAMA_HOST, OLLAMA_EMBEDDING_MODEL


class EmbeddingService:
    """Client for the Ollama embedding API."""

    def __init__(self):
        self.host = OLLAMA_HOST
        self.model = OLLAMA_EMBEDDING_MODEL
        self.client = Client(host=self.host)
        self.async_client = AsyncClient(host=self.host)

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text asynchronously (for query-time use)."""
        resp = await self.async_client.embeddings(
            model=self.model,
            prompt=text
        )
        return resp.get('embedding', [])

    def embed_text_sync(self, text: str) -> list[float]:
        """Embed a single text synchronously (for build_vectordb script)."""
        resp = self.client.embeddings(
            model=self.model,
            prompt=text
        )
        return resp.get('embedding', [])


embedding_service = EmbeddingService()

