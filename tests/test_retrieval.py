"""
Tests for EVIDRA Semantic Retrieval & Vector Indexing
Validates ChromaDB distance-to-similarity conversion and threshold boundaries.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import config
from services.ingest_service import ingest_document
from services.llm_service import get_embedding
from services.chroma_service import query_documents, delete_document_by_source


@pytest.fixture(scope="module")
def setup_test_doc():
    content = """# Page 1: Quantum Computing
Quantum supremacy represents algorithmic speedup through superposition and entanglement.
\x0c
# Page 2: Classical Limits
Classical supercomputers simulate 50 qubits with exponential memory penalties.
""".encode("utf-8")

    res = ingest_document(content, "quantum_benchmark.txt", overwrite=True)
    yield res
    delete_document_by_source("quantum_benchmark.txt")


def test_semantic_query_retrieval(setup_test_doc):
    q_emb = get_embedding("What is quantum supremacy?")
    hits = query_documents(q_emb, top_k=2)

    assert len(hits) >= 1
    top_hit = hits[0]
    # Cosine similarity score should exceed threshold
    assert top_hit["score"] >= config.SIMILARITY_THRESHOLD
    assert "superposition" in top_hit["text"].lower()
    assert top_hit["metadata"]["page_number"] == 1
    assert top_hit["metadata"]["source"] == "quantum_benchmark.txt"


def test_irrelevant_query_low_score(setup_test_doc):
    # Completely irrelevant query
    q_emb = get_embedding("What is the recipe for making Italian margherita pizza at home?")
    hits = query_documents(q_emb, top_k=2)

    assert len(hits) >= 1
    top_hit = hits[0]
    # Irrelevant query cosine similarity should fall below the relevance cutoff (0.55)
    assert top_hit["score"] < config.SIMILARITY_THRESHOLD


def test_threshold_boundary_filtering(setup_test_doc):
    q_rel = get_embedding("quantum superposition entanglement")
    hits_rel = query_documents(q_rel, top_k=2)
    valid_rel = [h for h in hits_rel if h["score"] >= config.SIMILARITY_THRESHOLD]
    assert len(valid_rel) >= 1

    q_irr = get_embedding("automobile carburetor fuel injection replacement")
    hits_irr = query_documents(q_irr, top_k=2)
    valid_irr = [h for h in hits_irr if h["score"] >= config.SIMILARITY_THRESHOLD]
    assert len(valid_irr) == 0


def test_retrieval_metadata_preservation(setup_test_doc):
    q_emb = get_embedding("classical simulation memory limits")
    hits = query_documents(q_emb, top_k=2)

    assert len(hits) >= 1
    top_hit = hits[0]
    meta = top_hit["metadata"]
    assert "document_id" in meta
    assert "chunk_id" in meta
    assert "page_number" in meta
    assert meta["page_number"] == 2
