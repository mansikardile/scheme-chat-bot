# SchemeSathi — Setup & Run Guide

## Prerequisites

- **Python** (any version — `uv` will auto-manage Python 3.14)
- **pip** (comes with Python)
- **Git** (to clone the repo)

---

## 1. Clone the Repository

```bash
git clone https://github.com/noventis-vit/scheme-sathi.git
cd scheme-sathi
```

---

## 2. Create the `.env` File

Create a file named `.env` in the **project root** (`scheme-sathi/.env`):

```env
# Ollama Configuration
OLLAMA_HOST="https://ai.11022006.xyz"
OLLAMA_EMBEDDING_MODEL="bge-m3"

# Embedding API
EMBEDDING_API_KEY="d932ee023c808f1f7b6bef3ff74589311da6e4415b26d7a100089d7624dd1b8d"

# Gemini
GEMINI_API_KEY="your-gemini-api-key-here"
```

> ⚠️ The `.env` file is git-ignored — never commit it.

---

## 3. Install `uv` (Package Manager)

```bash
pip install uv
```

---

## 4. Install Dependencies

Run this from the **project root** (`scheme-sathi/`):

```bash
uv sync
```

This will:
- Auto-download Python 3.14 (if not present)
- Create a `.venv` virtual environment
- Install all 92 dependencies from `uv.lock`

> ⏱️ First run takes ~2–3 minutes due to downloads.

---

## 5. Add the ChromaDB File

Place the `chroma.sqlite3` file (provided by your team) at:

```
scheme-sathi/
└── chroma_db/
    └── chroma.sqlite3   ← paste here
```

> If the folder doesn't exist, create it manually.

---

## 6. Run the Server

Run this from the **`src/`** subdirectory:

```bash
# Windows (PowerShell)
& "e:\path\to\scheme-sathi\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Or activate venv first, then run:
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

> **Important:** The command must be run from inside the `src/` folder, not the project root.

---

## 7. Access the App

| URL | Description |
|-----|-------------|
| http://localhost:8000 | Main frontend UI |
| http://localhost:8000/docs | Interactive API docs (Swagger) |
| http://localhost:8000/api/health | Health check — verify loaded schemes & vectors |

---

## Replacing the ChromaDB File

If you receive a new `chroma.sqlite3` from a teammate:

1. **Stop the server** (Ctrl+C in the terminal)
2. Wait a few seconds for the file lock to release
3. **Replace** `chroma_db/chroma.sqlite3` with the new file
4. **Restart** the server (repeat Step 6)

---

## Project Structure

```
scheme-sathi/
├── .env                  ← your secrets (not committed)
├── chroma_db/            ← vector database (not committed)
│   └── chroma.sqlite3
├── schemes/              ← raw scheme JSON files
├── src/
│   ├── backend/          ← FastAPI app
│   │   ├── main.py       ← server entry point
│   │   ├── config.py     ← reads .env variables
│   │   ├── rag_pipeline.py
│   │   ├── vector_store.py
│   │   └── gemini_service.py
│   └── frontend/         ← static HTML/CSS/JS
├── pyproject.toml        ← project & dependency config
└── uv.lock               ← locked dependency versions
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `uv` not found | Run `pip install uv` first |
| Port 8000 already in use | Change `--port 8000` to another port e.g. `--port 8001` |
| `chroma.sqlite3` locked | Stop the server fully before replacing the file |
| Gemini not responding | Check `GEMINI_API_KEY` is set correctly in `.env` |
| `ModuleNotFoundError: backend` | Make sure you're running from the `src/` directory |

& "e:\AI Engineer\Noventis-Scheme\scheme-sathi\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
