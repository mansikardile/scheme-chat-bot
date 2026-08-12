"""
Embedding service for SchemeSathi.
Wraps the bge-m3 embedding API for both async and sync usage.
"""

import httpx
from config import EMBEDDING_API_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL


class EmbeddingService:
    """Client for the bge-m3 embedding API."""

    def __init__(self):
        self.url = EMBEDDING_API_URL
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {EMBEDDING_API_KEY}',
        }
        self.model = EMBEDDING_MODEL

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text asynchronously (for query-time use)."""
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.url,
                headers=self.headers,
                json={'model': self.model, 'prompt': text},
            )
            resp.raise_for_status()
            return resp.json().get('embedding', [])

    def embed_text_sync(self, text: str) -> list[float]:
        """Embed a single text synchronously (for build_vectordb script)."""
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                self.url,
                headers=self.headers,
                json={'model': self.model, 'prompt': text},
            )
            resp.raise_for_status()
            return resp.json().get('embedding', [])


embedding_service = EmbeddingService()
