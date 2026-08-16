"""
Embedding service for SchemeSathi.
Wraps the local Ollama embedding API for both async and sync usage.
"""

from ollama import Client, AsyncClient
from backend.config import OLLAMA_HOST, OLLAMA_EMBEDDING_MODEL, EMBEDDING_API_KEY


class EmbeddingService:
    """Client for the Ollama embedding API."""

    def __init__(self):
        self.host = OLLAMA_HOST
        self.model = OLLAMA_EMBEDDING_MODEL
        
        headers = {}
        if EMBEDDING_API_KEY:
            headers["Authorization"] = f"Bearer {EMBEDDING_API_KEY}"
            
        self.client = Client(host=self.host, headers=headers)
        self.async_client = AsyncClient(host=self.host, headers=headers)

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

