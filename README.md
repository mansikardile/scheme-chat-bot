# Scheme Sathi

## Prerequisites
- [UV](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/download)

## Installation

Install dependencies:
```sh
uv sync
```

## Setup

### Scraping Schemes

The raw scheme data is already included in the `schemes/` directory.

To fetch newer/updated schemes:
```sh
uv run scraper
```

### Vector Database

The `chroma_db/` directory is omitted from git since it mostly consists of binary files.

To build or update the database:
```sh
uv run build-index
```

### Environment Variables

All variables below can be set in a `.env` file for defaults, or configured later from the web UI.

| Variable | Description |
|:--|:--|
|`GEMINI_API_KEY` | Gemini API key |
|`OLLAMA_HOST` | Host URL for Embedding model host |
|`OLLAMA_EMBEDDING_MODEL` | Embedding Model used. Currently the vector database has been trained using `bge-m3` |
|`EMBEDDING_API_KEY` | API key for remote embedding host |

## Running the Server

```sh
uv run server
```
The application will start on `http://localhost:8000`
