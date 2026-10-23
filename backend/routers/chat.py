"""
EVIDRA Chat Router — Strict Grounded RAG Endpoint
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from services.rag_service import evidra_query, Citation

router = APIRouter()


class ChatMessage(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2500, description="User question")
    history: List[ChatMessage] = Field(default_factory=list)
    language: str = Field("auto", description="'en', 'hi', or 'auto'")
    source_filter: Optional[str] = Field(None, description="Optional document source filename to filter search")
    top_k: Optional[int] = Field(None, ge=1, le=15, description="Number of passages to retrieve")
    strict_grounded: bool = Field(True, description="Enforce strict document grounding with canonical abstention")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override minimum similarity cutoff")


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    sources: List[str]
    language: str
    chunks_retrieved: int
    mode: str                           # grounded | abstained | blocked
    intent: Optional[str] = None
    sentiment: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Query the document knowledge vault with strict evidence grounding.
    Returns the generated answer along with structured citations (page numbers, quotes, scores).
    """
    try:
        history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
        result = evidra_query(
            query=req.query,
            history=history_dicts,
            top_k=req.top_k,
            language=req.language,
            source_filter=req.source_filter,
            strict_grounded=req.strict_grounded,
            similarity_threshold=req.similarity_threshold,
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
