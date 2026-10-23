"""
Unit and Integration Tests for EVIDRA Document Ingestion
"""

import pytest
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from services.ingest_service import clean_text, chunk_page_text, ingest_document
from services.chroma_service import list_documents, delete_document_by_source


def test_text_cleaning():
    raw = "This is a demon-\nstration of hyphenated words.\n\n\n\nExtra newlines."
    cleaned = clean_text(raw)
    assert "demonstration" in cleaned
    assert "\n\n\n" not in cleaned


def test_page_chunking_provenance():
    page_text = "Sentence one. Sentence two is longer. Sentence three is about policy regulations."
    chunks = chunk_page_text(page_text, page_number=4, chunk_size=10, overlap=2)
    assert len(chunks) >= 1
    for c in chunks:
        assert c["page_number"] == 4
        assert len(c["text"]) > 0


def test_empty_document_rejection():
    with pytest.raises(ValueError, match="empty or could not be read"):
        ingest_document(b"    \n   ", "empty_file.txt")


def test_unsupported_extension_rejection():
    with pytest.raises(ValueError, match="Unsupported file type"):
        ingest_document(b"binary content", "malicious_script.exe")


def test_ingest_multipage_txt():
    content = "This is page 1 introduction to the test document.\x0cThis is page 2 content about machine learning systems.".encode("utf-8")
    result = ingest_document(content, "test_multipage.txt", overwrite=True)
    assert result["total_pages"] == 2
    assert result["chunks_created"] >= 2
    assert "test_multipage.txt" in result["filename"]

    # Verify listing
    docs = list_documents()
    match = next((d for d in docs if d["source"] == "test_multipage.txt"), None)
    assert match is not None
    assert match["total_pages"] == 2

    # Clean up
    delete_document_by_source("test_multipage.txt")
