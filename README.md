# scheme-sathi

## Knowledge Base & RAG Layer

The backend includes a Knowledge Base and Retrieval-Augmented Generation (RAG) pipeline to enable conversational government scheme discovery.

### Components
- **Scheme Loader (`backend/scheme_loader.py`)**: Loads individual scraped scheme JSON files from `schemes/<slug>.json`.
- **Embedding Service (`backend/embedding_service.py`)**: Generates vector embeddings using the `bge-m3` model.
- **Vector Store (`backend/vector_store.py`)**: Stores scheme embeddings in a local persistent ChromaDB collection.
- **RAG Pipeline (`backend/rag_pipeline.py`)**: Handles multilingual query expansion, cosine similarity search, and grounding with Gemini LLM.
- **FastAPI Server (`backend/main.py`)**: Exposes REST endpoints for session management (`/api/chat/new`), RAG chat (`/api/chat`), and scheme details (`/api/schemes/{slug}`).

### Environment Setup
Create a `.env` file in the project root (see `backend/.env.example` for reference):

```env
GEMINI_API_KEY=your_gemini_api_key_here
EMBEDDING_API_URL=https://ai.11022006.xyz/api/embeddings
EMBEDDING_API_KEY=your_embedding_api_key_here
EMBEDDING_MODEL=bge-m3
```

### Building the Vector Database
> **Note:** `chroma_db/` is excluded from git tracking (`.gitignore`). The vector database must be built locally rather than committed.

1. Ensure scheme JSON files are present in the `schemes/` directory.
2. Run the build script:
   ```bash
   python backend/build_vectordb.py
   ```