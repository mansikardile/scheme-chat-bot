"""
ChromaDB vector store for SchemeSathi.
Backed by LangChain's Chroma integration; the raw chromadb collection is kept
accessible for the incremental-update operations in build_vectordb.py
(get/count/add/update/delete by ID) that LangChain's high-level wrapper does not expose.
"""

import chromadb
from langchain_chroma import Chroma
from backend.config import CHROMA_DB_PATH
from backend.embedding_service import embedding_service


class VectorStore:
    """LangChain-Chroma backed vector store for scheme storage and retrieval."""

    def __init__(self):
        self._chroma_client = None
        # LangChain Chroma wrapper — used for future retriever/chain integration
        self._lc_store: Chroma | None = None
        # Raw collection — used for batched add/update/delete/count/get operations
        self.collection = None

    def init(self):
        """Initialize the persistent ChromaDB client and LangChain Chroma wrapper."""
        self._chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        # Create or open the collection with cosine similarity
        self.collection = self._chroma_client.get_or_create_collection(
            name='schemes',
            metadata={'hnsw:space': 'cosine'},
        )

        # Wire up LangChain's Chroma wrapper sharing the same underlying client
        self._lc_store = Chroma(
            client=self._chroma_client,
            collection_name='schemes',
            embedding_function=embedding_service._lc_embeddings,
        )

        count = self.collection.count()
        print(f"ChromaDB (LangChain) initialized at {CHROMA_DB_PATH} — {count} vectors loaded.")

    # ------------------------------------------------------------------
    # Write operations — delegated to the raw collection so that the
    # exact batch add/update/delete interface used by build_vectordb.py
    # is preserved without change.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Search — uses the raw collection so we can pass a pre-computed
    # embedding (from our custom query-expansion step) directly.
    # Returns the same [{slug, document, metadata, distance}] shape as
    # before so rag_pipeline.py needs no changes.
    # ------------------------------------------------------------------

    def search(self, query_embedding: list[float], top_k: int = 15) -> list[dict]:
        """Search for the top-k most similar schemes using a pre-computed embedding."""
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
        """Delete and recreate the collection (used for full rebuild)."""
        if self._chroma_client:
            self._chroma_client.delete_collection('schemes')
            self.collection = self._chroma_client.get_or_create_collection(
                name='schemes',
                metadata={'hnsw:space': 'cosine'},
            )
            # Re-wire the LangChain wrapper to the fresh collection
            self._lc_store = Chroma(
                client=self._chroma_client,
                collection_name='schemes',
                embedding_function=embedding_service._lc_embeddings,
            )


vector_store = VectorStore()
