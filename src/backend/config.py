import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / '.env')

# Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# Embedding API (bge-m3)
EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL', 'https://ai.11022006.xyz/api/embeddings')
EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY', '')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'bge-m3')

# Ollama Configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
SERVER_OLLAMA_HOST = os.getenv('SERVER_OLLAMA_HOST', 'https://ai.11022006.xyz')
OLLAMA_LLM_MODEL = os.getenv('OLLAMA_LLM_MODEL', 'llama3')
OLLAMA_EMBEDDING_MODEL = os.getenv('OLLAMA_EMBEDDING_MODEL', 'bge-m3:latest')

# Available models configuration (User customizable)
# Format: {"id": "...", "name": "...", "provider": "gemini|ollama", "model_name": "..."}
CONFIGURED_MODELS = [
    {
        "id": "gemini-flash",
        "name": "Google Gemini Flash",
        "provider": "gemini",
        "model_name": "gemini-flash-latest",
        "description": "Google Cloud AI model"
    },
    {
        "id": "gemma3-4b",
        "name": "Gemma 3 (4B)",
        "provider": "ollama",
        "model_name": "gemma3:4b",
        "description": "Open Source Gemma 3 (4B)"
    },
]


# Paths
CHROMA_DB_PATH = str(ROOT_DIR / 'chroma_db')
SCHEMES_DIR = str(ROOT_DIR / 'schemes')  # produced by scraper/scrape_all.py, one JSON per slug
FRONTEND_DIR = str(ROOT_DIR / 'src' / 'frontend')

