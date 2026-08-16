# scheme-sathi

AI-powered assistant to help Indian citizens discover government welfare schemes.

## Backend Architecture

The backend is a **FastAPI** app with a RAG (Retrieval-Augmented Generation) pipeline:

| Component | Implementation |
|-----------|---------------|
| Embeddings | `langchain-ollama` → `OllamaEmbeddings` (bge-m3 via remote host) |
| Vector Store | `langchain-chroma` → `Chroma` backed by persistent ChromaDB |
| LLM | `langchain-google-genai` → `ChatGoogleGenerativeAI` (Gemini Flash) |
| Session History | `langchain-core` → `InMemoryChatMessageHistory` per session |
| Query Expansion | Custom Python (MULTILINGUAL_KEYWORD_MAP — domain logic, not replaced) |
| Card Selection | Custom Python (intent-gated — domain logic, not replaced) |

### LangChain Migration (branch: `refactor/langchain-migration`)

The backend was refactored from hand-rolled `ollama`, `google-genai`, and `chromadb` SDK calls
to the LangChain ecosystem. This is a **like-for-like** refactor — no new capabilities were added.
All three API endpoints (`POST /api/chat/new`, `POST /api/chat`, `GET /api/schemes/{slug}`) and
their request/response shapes are unchanged; the frontend needed zero edits.

### Legacy config entries (cleanup pending)

`config.py` still contains two entries that are **unused** since the Ollama embedding switch:

```python
EMBEDDING_API_URL  # was used by an older HTTP-based embedding client — now unused
EMBEDDING_API_KEY  # still forwarded as a Bearer header to Ollama — kept for auth
```

`EMBEDDING_API_URL` can be removed in a future cleanup PR once confirmed not referenced elsewhere.
`EMBEDDING_API_KEY` is still actively used (passed as an auth header to the remote Ollama host).

## Quick Start

See [SETUP.md](SETUP.md) for full setup instructions.