"""
EVIDRA Health & System Status Router
"""

from fastapi import APIRouter
from services.chroma_service import collection_count
import config

router = APIRouter()


@router.get("/health")
async def health():
    """System health check and configuration summary."""
    return {
        "status": "online",
        "system": "EVIDRA — Evidence-Grounded Document Intelligence",
        "version": "2.0.0",
        "llm_provider": "Groq",
        "llm_model": config.GROQ_CHAT_MODEL,
        "fallback_model": config.GROQ_FALLBACK_MODEL,
        "embedding_provider": "Google Gemini",
        "embedding_model": config.GEMINI_EMBED_MODEL,
        "embedding_dimension": config.EMBED_DIMENSION,
        "vector_store": "ChromaDB (Persistent Cosine)",
        "indexed_chunks": collection_count(),
        "strict_grounding": config.STRICT_GROUNDED_MODE,
        "similarity_threshold": config.SIMILARITY_THRESHOLD,
    }
