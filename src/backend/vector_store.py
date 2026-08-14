"""
ChromaDB vector store for SchemeSathi.
Provides persistent storage and cosine similarity search over scheme embeddings.
"""

import chromadb
from backend.config import CHROMA_DB_PATH


class VectorStore:
    """Wrapper around ChromaDB for scheme vector storage and retrieval."""

    def __init__(self):
        self.client = None
        self.collection = None

    def init(self):
        """Initialize the persistent ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name='schemes',
            metadata={'hnsw:space': 'cosine'},
        )
        count = self.collection.count()
        print(f"ChromaDB initialized at {CHROMA_DB_PATH} — {count} vectors loaded.")

    def add_schemes(self, ids, embeddings, documents, metadatas):
        """Add scheme embeddings in batches."""
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

    def update_schemes(self, ids, embeddings, documents, metadatas):
        """Update scheme embeddings in batches."""
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.update(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

    def delete_schemes(self, ids):
        """Delete schemes by ID."""
        if not ids:
            return
        self.collection.delete(ids=ids)

    def search(self, query_embedding, top_k=15):
        """Search for the top-k most similar schemes."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances'],
        )

        schemes = []
        if results and results['ids'] and results['ids'][0]:
            for i, slug in enumerate(results['ids'][0]):
                schemes.append({
                    'slug': slug,
                    'document': results['documents'][0][i] if results['documents'] else '',
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0,
                })
        return schemes

    def clear(self):
        """Delete and recreate the collection."""
        if self.client:
            self.client.delete_collection('schemes')
            self.collection = self.client.get_or_create_collection(
                name='schemes',
                metadata={'hnsw:space': 'cosine'},
            )


vector_store = VectorStore()
