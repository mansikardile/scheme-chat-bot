import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / '.env')

# Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# Embedding API (bge-m3)
EMBEDDING_API_URL = os.getenv('EMBEDDING_API_URL', 'https://ai.11022006.xyz/api/embeddings')
EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY', '')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'bge-m3')

# Paths
CHROMA_DB_PATH = str(ROOT_DIR / 'chroma_db')
SCHEMES_DIR = str(ROOT_DIR / 'schemes')  # produced by scraper/scrape_all.py, one JSON per slug
FRONTEND_DIR = str(ROOT_DIR / 'frontend')
