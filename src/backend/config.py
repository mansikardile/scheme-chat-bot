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

# Default built-in models
CONFIGURED_MODELS = [
    {
        "id": "gemini-flash",
        "name": "Google Gemini",
        "provider": "gemini",
        "model_name": "gemini-3.5-flash-lite",
        "is_local": False,
        "description": "Google Cloud AI model"
    },
]

# Paths
CHROMA_DB_PATH = str(ROOT_DIR / 'chroma_db')
SCHEMES_DIR = str(ROOT_DIR / 'schemes')
SCHEMES_DB_PATH = str(ROOT_DIR / 'schemes.duckdb')
FRONTEND_DIR = str(ROOT_DIR / 'src' / 'frontend')
