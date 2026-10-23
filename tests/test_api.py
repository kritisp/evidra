"""
API Endpoint Integration Tests
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert "Groq" in data["llm_provider"]
    assert "Gemini" in data["embedding_provider"]
    assert data["embedding_dimension"] == 3072


def test_documents_listing():
    resp = client.get("/api/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert "documents" in data
    assert "total_chunks" in data


def test_document_upload_and_delete():
    file_bytes = b"Page 1: Temporary test doc for API integration testing.\n"
    files = {"file": ("api_test_doc.txt", file_bytes, "text/plain")}
    resp = client.post("/api/documents/upload", files=files, data={"overwrite": "true"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunks_created"] >= 1

    # Delete doc
    del_resp = client.delete("/api/documents/api_test_doc.txt")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "success"


def test_chat_endpoint_validation():
    # Empty query should fail validation
    resp = client.post("/api/chat", json={"query": ""})
    assert resp.status_code == 422
