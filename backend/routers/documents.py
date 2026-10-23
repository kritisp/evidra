"""
EVIDRA Document Router — Document Ingestion & Provenance Management
"""

import os
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel

from services.ingest_service import ingest_document
from services.chroma_service import (
    list_documents,
    delete_document_by_source,
    collection_count,
    reset_collection,
)
import config

router = APIRouter()


class DocumentInfo(BaseModel):
    document_id: str
    source: str
    language: str
    file_type: str
    chunks: int
    total_pages: int
    pages: List[int]
    uploaded_at: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_documents: int
    total_chunks: int


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    overwrite: bool = Form(True),
):
    """
    Upload and ingest a document (PDF, TXT, MD, DOCX, CSV) with page-level provenance.
    """
    filename = file.filename or "uploaded_document"
    ext = os.path.splitext(filename)[-1].lower()

    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported: {sorted(list(config.ALLOWED_EXTENSIONS))}"
        )

    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is completely empty.")

        result = ingest_document(
            file_data=data,
            filename=filename,
            language=language,
            overwrite=overwrite,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def get_documents():
    """List all indexed documents, including page count and chunk breakdown."""
    docs = list_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total_documents=len(docs),
        total_chunks=collection_count(),
    )


@router.delete("/documents/{filename:path}")
async def delete_document(filename: str):
    """Delete a document and all associated chunks from the vector database."""
    deleted = delete_document_by_source(filename)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
    return {
        "status": "success",
        "message": f"Successfully deleted '{filename}'",
        "chunks_deleted": deleted,
    }


@router.post("/documents/reset")
async def reset_all_documents():
    """Reset the entire document collection."""
    reset_collection()
    return {"status": "success", "message": "All documents and vector indices cleared."}
