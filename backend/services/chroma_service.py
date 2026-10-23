"""
EVIDRA Vector Database Service (ChromaDB)
- Persistent local / disk storage
- Cosine distance metric
- Strict metadata preservation: document_id, source, page_number, chunk_id, etc.

Mathematical Distance Semantics:
  ChromaDB with 'hnsw:space: cosine' computes Cosine Distance:
      d = 1.0 - cos_sim(u, v)
  where u, v are unit-normalized embedding vectors.
  - Identical vectors:      d = 0.0  => cos_sim = 1.0
  - Orthogonal (unrelated): d = 1.0  => cos_sim = 0.0
  - Opposite vectors:       d = 2.0  => cos_sim = -1.0

  We compute raw Cosine Similarity:
      cosine_similarity = 1.0 - distance
  For semantic document retrieval with Gemini embeddings:
  - Strong semantic matches: cos_sim >= 0.60
  - Boundary relevance cutoff: cos_sim >= 0.55
  - Irrelevant/out-of-corpus queries: cos_sim <= 0.50
"""

import uuid
from typing import Optional, List, Dict, Any
import chromadb
from chromadb.config import Settings
import config

_client: Optional[chromadb.PersistentClient] = None
_collection = None


def init_chroma():
    global _client, _collection
    _client = chromadb.PersistentClient(
        path=config.CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    _collection = _client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def get_collection():
    global _collection
    if _collection is None:
        init_chroma()
    return _collection


def add_documents(
    texts: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
    ids: Optional[List[str]] = None,
) -> List[str]:
    """Insert text chunks, embeddings, and provenance metadata into ChromaDB."""
    col = get_collection()
    if ids is None:
        ids = [m.get("chunk_id", str(uuid.uuid4())) for m in metadatas]

    col.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    return ids


def query_documents(
    query_embedding: List[float],
    top_k: Optional[int] = None,
    filter_meta: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Search ChromaDB by cosine distance.
    Returns ranked list of chunks with complete provenance metadata, raw cosine distance,
    and cosine similarity score (score = 1.0 - distance).
    """
    col = get_collection()
    count = col.count()
    if count == 0:
        return []

    k = min(top_k or config.TOP_K_RESULTS, count)
    kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if filter_meta:
        kwargs["where"] = filter_meta

    results = col.query(**kwargs)

    output = []
    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    ids = results.get("ids", [[]])[0]

    for i in range(len(docs)):
        dist = float(dists[i])
        # Raw cosine similarity: 1.0 - distance
        raw_similarity = round(1.0 - dist, 4)
        output.append({
            "id": ids[i] if i < len(ids) else f"chunk_{i}",
            "text": docs[i],
            "metadata": metas[i],
            "distance": round(dist, 4),
            "score": raw_similarity,  # Cosine similarity in [-1.0, 1.0]
        })

    # Sort descending by cosine similarity score
    output.sort(key=lambda x: x["score"], reverse=True)
    return output


def delete_document_by_source(source: str) -> int:
    """Delete all chunks belonging to a document source."""
    col = get_collection()
    results = col.get(where={"source": source}, include=["metadatas"])
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        col.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def list_documents() -> List[Dict[str, Any]]:
    """Return unique documents with total chunks, page count, and upload timestamp."""
    col = get_collection()
    if col.count() == 0:
        return []

    all_data = col.get(include=["metadatas"])
    metas = all_data.get("metadatas", [])
    doc_map: Dict[str, Dict[str, Any]] = {}

    for m in metas:
        src = m.get("source", "unknown")
        page = m.get("page_number", 1)
        if src not in doc_map:
            doc_map[src] = {
                "document_id": m.get("document_id", src),
                "source": src,
                "language": m.get("language", "unknown"),
                "file_type": m.get("file_type", ""),
                "chunks": 0,
                "pages": set(),
                "uploaded_at": m.get("uploaded_at", ""),
            }
        doc_map[src]["chunks"] += 1
        doc_map[src]["pages"].add(page)

    result = []
    for d in doc_map.values():
        d["total_pages"] = len(d["pages"])
        d["pages"] = sorted(list(d["pages"]))
        result.append(d)

    return result


def collection_count() -> int:
    return get_collection().count()


def reset_collection() -> None:
    """Clear all documents in the collection (useful for clean evaluations and tests)."""
    global _collection
    if _client is not None:
        try:
            _client.delete_collection(config.CHROMA_COLLECTION)
        except Exception:
            pass
        _collection = None
        init_chroma()
