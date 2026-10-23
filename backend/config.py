import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from current directory or backend directory
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# ==============================================================================
# 1. GROQ LLM CONFIGURATION (Generation)
# ==============================================================================
_groq_raw = [
    os.getenv("GROQ_API_KEY_1", ""),
    os.getenv("GROQ_API_KEY_2", ""),
    os.getenv("GROQ_API_KEY_3", ""),
    os.getenv("GROQ_API_KEY_4", ""),
    os.getenv("GROQ_API_KEY_5", ""),
]
GROQ_API_KEYS = [k.strip() for k in _groq_raw if k.strip()]

GROQ_CHAT_MODEL     = os.getenv("GROQ_CHAT_MODEL", "qwen/qwen3.8-27b")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b")

# ==============================================================================
# 2. GOOGLE GEMINI CONFIGURATION (Embeddings)
# ==============================================================================
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_EMBED_MODEL  = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")
EMBED_DIMENSION     = int(os.getenv("EMBED_DIMENSION", "3072"))

# ==============================================================================
# 3. CHROMADB VECTOR STORE
# ==============================================================================
_raw_persist = os.getenv("CHROMA_PERSIST_DIR", "")
if _raw_persist and os.path.isabs(_raw_persist):
    CHROMA_PERSIST_DIR = _raw_persist
else:
    # Anchor relative path to project root so backend and root scripts share the same store
    _project_root = Path(__file__).resolve().parent.parent
    CHROMA_PERSIST_DIR = str(_project_root / (_raw_persist.lstrip("./") if _raw_persist else "chroma_db"))

CHROMA_COLLECTION   = os.getenv("CHROMA_COLLECTION", "evidra_documents")

# ==============================================================================
# 4. RAG PIPELINE & GROUNDING
# ==============================================================================
CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K_RESULTS        = int(os.getenv("TOP_K_RESULTS", "5"))
# Cosine similarity cutoff (S = 1.0 - distance). Relevant >= 0.55, Irrelevant <= 0.50
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
STRICT_GROUNDED_MODE = os.getenv("STRICT_GROUNDED_MODE", "true").lower() in ("1", "true", "yes")

# Document constraints
MAX_FILE_SIZE_MB     = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
ALLOWED_EXTENSIONS   = {".pdf", ".txt", ".md", ".docx", ".csv"}

# Reliability & retry
RETRY_ATTEMPTS       = int(os.getenv("RETRY_ATTEMPTS", "3"))
RETRY_DELAY_SECONDS  = float(os.getenv("RETRY_DELAY_SECONDS", "1.0"))

SUPPORTED_LANGUAGES  = ["en", "hi", "auto"]
