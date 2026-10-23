"""
EVIDRA Ingestion Service — Page-Aware Document Parsing & Provenance Chunking
Pipeline:
  1. Parse file page-by-page (PDF, DOCX, TXT, MD, CSV)
  2. Clean and normalize text
  3. Chunk intelligently while preserving exact page provenance
  4. Generate embeddings via Gemini (models/gemini-embedding-001)
  5. Store chunks with rich metadata in ChromaDB
"""

import io
import os
import re
import uuid
import datetime
from typing import List, Dict, Any, Optional

import config
from services.llm_service import get_embeddings_batch
from services.chroma_service import add_documents, delete_document_by_source

# ── Language detection ────────────────────────────────────────────────────────
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
_HINGLISH_WORDS = {
    "kya","kyun","kaise","kaisa","kaisi","kab","kahan","kaun","kitna","kitne",
    "aap","tum","main","hum","woh","yeh","ye","vo","kar","karo","kare",
    "hain","hai","tha","thi","hoga","hogi","honge","nahi","nhi","mujhe",
    "tumhe","use","inhe","unhe","aur","ya","lekin","magar","phir","ab",
    "accha","theek","sahi","galat","batao","help","madad","chahiye"
}


def detect_language(text: str) -> str:
    """Detect whether query/document is primarily Hindi or English."""
    if not text:
        return "en"
    deva_count = len(_DEVANAGARI_RE.findall(text))
    if deva_count / max(len(text), 1) > 0.05:
        return "hi"
    tokens = re.findall(r'[a-zA-Z]+', text.lower())
    if not tokens:
        return "en"
    matches = sum(1 for t in tokens if t in _HINGLISH_WORDS)
    threshold = 1 if len(tokens) <= 4 else 2
    if matches >= threshold:
        return "hi"
    return "en"


# ── Text Cleaning & Normalization ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Clean and normalize extracted document text."""
    if not text:
        return ""
    # Repair hyphenated word wraps: "implemen-\ntation" -> "implementation"
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Replace non-breaking spaces and unusual whitespace
    text = text.replace('\xa0', ' ').replace('\r\n', '\n').replace('\r', '\n')
    # Collapse multiple consecutive newlines (3+ -> 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# ── Page-Aware File Parsers ───────────────────────────────────────────────────
def _parse_pdf_pages(data: bytes) -> List[Dict[str, Any]]:
    """Extract text from PDF page by page using pypdf with fallback."""
    pages = []
    # Primary parser: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        for idx, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = clean_text(raw)
            if cleaned:
                pages.append({"page_number": idx, "text": cleaned})
        if pages:
            return pages
    except Exception as e:
        print(f"pypdf extraction notice: {e}, attempting pdfplumber fallback...")

    # Secondary parser: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                cleaned = clean_text(raw)
                if cleaned:
                    pages.append({"page_number": idx, "text": cleaned})
        return pages
    except ImportError:
        pass

    if not pages:
        raise ValueError("Could not extract any readable text from this PDF.")
    return pages


def _parse_txt_pages(data: bytes) -> List[Dict[str, Any]]:
    text = clean_text(data.decode("utf-8", errors="replace"))
    if not text:
        return []
    # If the text has form-feed page markers (\x0c), split on them
    raw_pages = text.split("\x0c")
    if len(raw_pages) > 1:
        return [{"page_number": i, "text": clean_text(p)} for i, p in enumerate(raw_pages, start=1) if clean_text(p)]
    return [{"page_number": 1, "text": text}]


def _parse_md_pages(data: bytes) -> List[Dict[str, Any]]:
    text = clean_text(data.decode("utf-8", errors="replace"))
    if not text:
        return []
    # Section-aware splitting on major headers `# `
    sections = re.split(r'(?=\n#+ )', text)
    if len(sections) > 1:
        pages = []
        for i, s in enumerate(sections, start=1):
            s_clean = clean_text(s)
            if s_clean:
                pages.append({"page_number": i, "text": s_clean})
        return pages
    return [{"page_number": 1, "text": text}]


def _parse_docx_pages(data: bytes) -> List[Dict[str, Any]]:
    try:
        from docx import Document
        doc = Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = clean_text("\n\n".join(paragraphs))
        return [{"page_number": 1, "text": text}] if text else []
    except Exception as e:
        raise ValueError(f"Failed to parse DOCX file: {e}")


def _parse_csv_pages(data: bytes) -> List[Dict[str, Any]]:
    import csv
    lines = data.decode("utf-8", errors="replace").splitlines()
    reader = csv.reader(lines)
    rows = [", ".join(row) for row in reader if any(row)]
    text = clean_text("\n".join(rows))
    return [{"page_number": 1, "text": text}] if text else []


PARSERS = {
    ".pdf":  _parse_pdf_pages,
    ".txt":  _parse_txt_pages,
    ".md":   _parse_md_pages,
    ".docx": _parse_docx_pages,
    ".csv":  _parse_csv_pages,
}


# ── Intelligent Page-Aware Chunking ───────────────────────────────────────────
def chunk_page_text(
    text: str,
    page_number: int,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Split a single page's text into overlapping, sentence-aware chunks while strictly
    preserving page provenance.
    """
    if not text or len(text.strip()) < 15:
        return []

    # Split into sentences (Devanagari danda '।', question mark, exclamation, or period)
    sentences = re.split(r'(?<=[।?.!])\s+', text.strip())
    chunks = []
    current_words: List[str] = []

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        if len(current_words) + len(words) > chunk_size and current_words:
            chunk_body = " ".join(current_words)
            if len(chunk_body.strip()) > 20:
                chunks.append(chunk_body)
            # Retain overlap words
            current_words = current_words[-overlap:] if overlap > 0 else []
        current_words.extend(words)

    if current_words:
        chunk_body = " ".join(current_words)
        if len(chunk_body.strip()) > 20:
            chunks.append(chunk_body)

    return [{"page_number": page_number, "text": c} for c in chunks]


# ── Full Ingestion Pipeline ───────────────────────────────────────────────────
def ingest_document(
    file_data: bytes,
    filename: str,
    language: str = "auto",
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Parse document, extract pages, chunk intelligently, compute Gemini embeddings,
    and persist into ChromaDB with complete page-level provenance metadata.
    """
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in PARSERS:
        raise ValueError(f"Unsupported file type: {ext}. Allowed: {list(PARSERS.keys())}")

    file_size_mb = len(file_data) / (1024 * 1024)
    if file_size_mb > config.MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large: {file_size_mb:.1f}MB (max {config.MAX_FILE_SIZE_MB}MB)")

    # 1. Parse pages
    pages = PARSERS[ext](file_data)
    if not pages:
        raise ValueError(f"Document '{filename}' appears empty or could not be read.")

    full_text = " ".join(p["text"] for p in pages)
    detected_lang = detect_language(full_text) if language == "auto" else language

    # 2. Chunk page by page
    all_chunks: List[Dict[str, Any]] = []
    for page in pages:
        page_chunks = chunk_page_text(
            text=page["text"],
            page_number=page["page_number"],
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )
        all_chunks.extend(page_chunks)

    if not all_chunks:
        raise ValueError(f"No usable text chunks could be extracted from '{filename}'.")

    # 3. Clean up previous version if overwriting
    if overwrite:
        deleted = delete_document_by_source(filename)
        if deleted > 0:
            print(f"🗑️ Replaced {deleted} old chunks for '{filename}'")

    # 4. Generate embeddings (batched via Gemini)
    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = get_embeddings_batch(chunk_texts)

    # 5. Build provenance metadata
    now = datetime.datetime.utcnow().isoformat()
    doc_id = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)
    metadatas: List[Dict[str, Any]] = []
    chunk_ids: List[str] = []

    for idx, c in enumerate(all_chunks):
        p_num = c["page_number"]
        cid = f"{doc_id}_p{p_num}_c{idx + 1}"
        chunk_ids.append(cid)
        metadatas.append({
            "document_id": doc_id,
            "source": filename,
            "page_number": p_num,
            "page_label": f"Page {p_num}",
            "chunk_id": cid,
            "chunk_index": idx + 1,
            "total_chunks": len(all_chunks),
            "char_count": len(c["text"]),
            "language": detected_lang,
            "uploaded_at": now,
            "file_type": ext,
        })

    # 6. Store in ChromaDB
    inserted_ids = add_documents(
        texts=chunk_texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=chunk_ids,
    )

    return {
        "filename": filename,
        "document_id": doc_id,
        "total_pages": len(pages),
        "chunks_created": len(all_chunks),
        "language": detected_lang,
        "characters": len(full_text),
        "sample_ids": inserted_ids[:3],
        "message": f"Successfully indexed '{filename}' ({len(pages)} pages, {len(all_chunks)} chunks).",
    }
