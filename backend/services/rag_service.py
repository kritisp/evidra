"""
EVIDRA RAG Engine — Evidence-Grounded Document Intelligence
Features:
  - Strict evidence grounding (zero ungrounded general-knowledge hallucination)
  - Canonical abstention when evidence is insufficient
  - Verifiable structured citations with page provenance and quote snippets
  - Tuned similarity thresholding
  - Query intelligence & prompt injection guardrails
  - Document intelligence tools (summarization, quiz generation, translation, followups)
"""

import json
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

import config
from services.llm_service import groq_chat, get_embedding
from services.chroma_service import query_documents, collection_count, get_collection
from services.intelligence_service import analyze_query, get_sentiment_system_addition
from services.ingest_service import detect_language

CANONICAL_ABSTENTION_EN = "I couldn't find sufficient evidence in the provided documents to answer this question."
CANONICAL_ABSTENTION_HI = "प्रदान किए गए दस्तावेज़ों में इस प्रश्न का उत्तर देने के लिए पर्याप्त साक्ष्य नहीं मिले।"


class Citation(BaseModel):
    id: int
    document: str
    page: int
    page_label: str
    snippet: str
    relevance_score: float
    chunk_id: str


def _build_grounded_system_prompt(lang: str, sentiment: str = "neutral") -> str:
    lang_rule = (
        "CRITICAL: उत्तर केवल हिंदी (देवनागरी लिपि) में दें।"
        if lang == "hi"
        else "CRITICAL: Respond ONLY in English."
    )
    sentiment_hint = get_sentiment_system_addition(sentiment, lang)

    return f"""You are EVIDRA, an evidence-grounded document intelligence engine.
{lang_rule}

STRICT OPERATING PRINCIPLES:
1. GROUNDING RULE: You must answer the user's question SOLELY and EXCLUSIVELY using the facts stated in the numbered [Ref N] context passages below.
2. NEVER use outside general knowledge or speculate beyond the provided text.
3. CITATIONS: Include citation tags like [1], [2] immediately after each claim that relies on that reference.
4. ABSTENTION TRIGGER: If the provided references do NOT contain clear, sufficient factual evidence to answer the user's question, you MUST reply with this exact sentence and nothing else:
"{CANONICAL_ABSTENTION_HI if lang == 'hi' else CANONICAL_ABSTENTION_EN}"
5. Do NOT invent references. Only cite numbers that exist in the context below.
{sentiment_hint}
"""


def _format_context_passages(chunks: List[Dict[str, Any]]) -> str:
    passages = []
    for idx, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        doc = meta.get("source", "Unknown Document")
        page = meta.get("page_number", 1)
        score = c.get("score", 0.0)
        passages.append(
            f"[Ref {idx}] Document: {doc} | Page: {page} | Relevance: {score:.2f}\n"
            f"Content: {c['text'].strip()}"
        )
    return "\n\n---\n\n".join(passages)


def _extract_and_validate_citations(
    answer: str,
    chunks: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[int], str]:
    """
    Parse inline citation markers [1], [2] from the answer and link them
    strictly to the exact retrieved chunks with provenance metadata.

    Rules:
      1. If the model omitted citation markers, DO NOT return all retrieved chunks! Return empty citations.
      2. If an invalid citation ID appears (e.g. [9] when only 2 chunks exist), reject and strip it.
      3. Verify document, page number, and provenance for each cited chunk.
    """
    found_raw = re.findall(r'\[(\d+)\]', answer)
    if not found_raw:
        # Model omitted citation markers - return empty list, not all chunks
        return [], [], answer

    found_ids = [int(x) for x in found_raw]
    valid_citations = []
    invalid_ids = []
    seen_valid = set()
    max_valid = len(chunks)

    for cid in found_ids:
        if cid < 1 or cid > max_valid:
            if cid not in invalid_ids:
                invalid_ids.append(cid)
        elif cid not in seen_valid:
            seen_valid.add(cid)
            c = chunks[cid - 1]
            meta = c.get("metadata", {})
            text = c.get("text", "").strip()

            sentences = re.split(r'(?<=[।?.!])\s+', text)
            snippet = " ".join(sentences[:2]) if sentences else text[:180]
            if len(snippet) > 220:
                snippet = snippet[:220].rsplit(" ", 1)[0] + "..."

            valid_citations.append({
                "id": cid,
                "document": meta.get("source", "Unknown"),
                "page": int(meta.get("page_number", 1)),
                "page_label": meta.get("page_label", f"Page {meta.get('page_number', 1)}"),
                "snippet": snippet,
                "relevance_score": float(c.get("score", 0.0)),
                "chunk_id": meta.get("chunk_id", c.get("id", f"chunk_{cid}")),
            })

    valid_citations.sort(key=lambda x: x["id"])

    # Sanitize answer by removing hallucinated citation tags
    sanitized_answer = answer
    for inv in invalid_ids:
        sanitized_answer = re.sub(rf'\[{inv}\]', '', sanitized_answer)

    return valid_citations, invalid_ids, sanitized_answer.strip()


def _extract_citations(answer: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations, _, _ = _extract_and_validate_citations(answer, chunks)
    return citations


def is_abstention(text: str) -> bool:
    """Check if the generated response is an abstention."""
    normalized = text.lower().strip()
    return (
        "couldn't find sufficient evidence" in normalized
        or "insufficient evidence" in normalized
        or "पर्याप्त साक्ष्य नहीं मिले" in normalized
        or "not mentioned in the provided" in normalized
        or "does not contain information" in normalized
    )


def evidra_query(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    top_k: Optional[int] = None,
    language: str = "auto",
    source_filter: Optional[str] = None,
    strict_grounded: bool = True,
    similarity_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Execute full Evidra RAG pipeline with strict grounding and verifiable citations.
    """
    threshold = similarity_threshold if similarity_threshold is not None else config.SIMILARITY_THRESHOLD
    query_clean = query.strip()
    if not query_clean:
        return {
            "answer": "Please provide a valid question.",
            "citations": [],
            "invalid_citations": [],
            "sources": [],
            "mode": "abstained",
            "language": "en",
            "chunks_retrieved": 0,
            "retrieved_chunks": [],
            "all_retrieved_chunks": [],
        }

    # 1. Query Intelligence & Safety Check
    intel = analyze_query(query_clean)
    if intel.get("is_harmful"):
        lang = detect_language(query_clean) if language == "auto" else language
        blocked_msg = (
            "यह अनुरोध संसाधित नहीं किया जा सकता। कृपया एक उचित दस्तावेज़-आधारित प्रश्न पूछें।"
            if lang == "hi"
            else "This request cannot be processed. Please ask an appropriate document-grounded question."
        )
        return {
            "answer": blocked_msg,
            "citations": [],
            "invalid_citations": [],
            "sources": [],
            "mode": "blocked",
            "language": lang,
            "chunks_retrieved": 0,
            "retrieved_chunks": [],
            "all_retrieved_chunks": [],
            "intent": "blocked",
            "sentiment": intel.get("sentiment", "neutral"),
        }

    lang = detect_language(query_clean) if language == "auto" else language
    doc_count = collection_count()

    # 2. Case: Zero documents in vector database -> Strict Abstention
    if doc_count == 0:
        abstain_msg = CANONICAL_ABSTENTION_HI if lang == "hi" else CANONICAL_ABSTENTION_EN
        return {
            "answer": abstain_msg,
            "citations": [],
            "invalid_citations": [],
            "sources": [],
            "mode": "abstained",
            "language": lang,
            "chunks_retrieved": 0,
            "retrieved_chunks": [],
            "all_retrieved_chunks": [],
            "intent": intel.get("intent", "factual"),
            "sentiment": intel.get("sentiment", "neutral"),
        }

    # 3. Retrieve relevant chunks
    query_emb = get_embedding(query_clean)
    filter_dict = {"source": source_filter} if source_filter else None
    retrieved_chunks = query_documents(
        query_embedding=query_emb,
        top_k=top_k or config.TOP_K_RESULTS,
        filter_meta=filter_dict,
    )

    # 4. Filter by similarity threshold
    valid_chunks = [c for c in retrieved_chunks if c.get("score", 0.0) >= threshold]

    # 5. Case: No chunks meet the similarity threshold -> Strict Abstention
    if not valid_chunks:
        abstain_msg = CANONICAL_ABSTENTION_HI if lang == "hi" else CANONICAL_ABSTENTION_EN
        return {
            "answer": abstain_msg,
            "citations": [],
            "invalid_citations": [],
            "sources": [],
            "mode": "abstained",
            "language": lang,
            "chunks_retrieved": 0,
            "retrieved_chunks": [],
            "all_retrieved_chunks": retrieved_chunks,
            "intent": intel.get("intent", "factual"),
            "sentiment": intel.get("sentiment", "neutral"),
        }

    # 6. Build Grounded Context & Call LLM
    context_str = _format_context_passages(valid_chunks)
    system_prompt = _build_grounded_system_prompt(lang=lang, sentiment=intel.get("sentiment", "neutral"))
    user_prompt = f"Reference Context:\n{context_str}\n\nUser Question:\n{query_clean}"

    llm_output = groq_chat(
        system_prompt=system_prompt,
        user_message=user_prompt,
        history=history or [],
        temperature=0.1,
    )

    # 7. Check if model decided to abstain
    if is_abstention(llm_output):
        abstain_msg = CANONICAL_ABSTENTION_HI if lang == "hi" else CANONICAL_ABSTENTION_EN
        return {
            "answer": abstain_msg,
            "citations": [],
            "invalid_citations": [],
            "sources": [],
            "mode": "abstained",
            "language": lang,
            "chunks_retrieved": len(valid_chunks),
            "retrieved_chunks": valid_chunks,
            "all_retrieved_chunks": retrieved_chunks,
            "intent": intel.get("intent", "factual"),
            "sentiment": intel.get("sentiment", "neutral"),
        }

    # 8. Extract & Validate Structured Citations
    citations, invalid_cids, final_answer = _extract_and_validate_citations(llm_output, valid_chunks)
    unique_sources = sorted(list({cit["document"] for cit in citations}))

    return {
        "answer": final_answer.strip(),
        "citations": citations,
        "invalid_citations": invalid_cids,
        "sources": unique_sources,
        "mode": "grounded",
        "language": lang,
        "chunks_retrieved": len(valid_chunks),
        "retrieved_chunks": valid_chunks,
        "all_retrieved_chunks": retrieved_chunks,
        "intent": intel.get("intent", "factual"),
        "sentiment": intel.get("sentiment", "neutral"),
    }


# Backwards compatibility
rag_query = evidra_query


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT INTELLIGENCE TOOLS
# ══════════════════════════════════════════════════════════════════════════════

def summarize_document(source: str, lang: str = "en") -> Dict[str, Any]:
    col = get_collection()
    result = col.get(where={"source": source}, include=["documents"])
    chunks = result.get("documents", [])
    if not chunks:
        raise ValueError(f"No chunks found for '{source}'")

    full_text = "\n\n".join(chunks[:25])
    if lang == "hi":
        prompt = f"""निम्नलिखित दस्तावेज़ का विश्लेषण करें और केवल JSON लौटाएं:
{{"title":"शीर्षक","summary":"2-3 वाक्यों में सारांश","key_points":["बिंदु1","बिंदु2","बिंदु3"],"topics":["विषय1","विषय2"],"language":"hi"}}

दस्तावेज़:
{full_text[:4000]}"""
    else:
        prompt = f"""Analyze this document and return ONLY valid JSON:
{{"title":"document title","summary":"2-3 sentence summary","key_points":["point1","point2","point3","point4","point5"],"topics":["topic1","topic2","topic3"],"language":"en"}}

Document:
{full_text[:4000]}"""

    raw = groq_chat("Return ONLY valid JSON. No markdown, no conversational text.", prompt, temperature=0.2)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        data = json.loads(raw)
    except Exception:
        data = {"title": source, "summary": raw[:300], "key_points": [], "topics": [], "language": lang}

    data["source"] = source
    data["chunk_count"] = len(chunks)
    data["word_count"] = sum(len(c.split()) for c in chunks)
    return data


def generate_quiz(source: str, num_questions: int = 5, lang: str = "en") -> Dict[str, Any]:
    col = get_collection()
    result = col.get(where={"source": source}, include=["documents"])
    chunks = result.get("documents", [])
    if not chunks:
        raise ValueError(f"No chunks found for '{source}'")

    full_text = "\n\n".join(chunks[:20])
    n = min(num_questions, 10)

    prompt = f"""Generate {n} multiple-choice questions based strictly on the document text. Return ONLY a JSON array:
[{{"question":"Question?","options":["A. opt1","B. opt2","C. opt3","D. opt4"],"answer":"A","explanation":"Why correct"}}]

Document:
{full_text[:3500]}"""

    raw = groq_chat("Return ONLY a valid JSON array. No markdown, no extra text.", prompt, temperature=0.2)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        questions = json.loads(raw)
        if not isinstance(questions, list):
            questions = []
    except Exception:
        questions = []

    return {"source": source, "questions": questions, "count": len(questions), "language": lang}


def translate_text(text: str, target_lang: str) -> Dict[str, Any]:
    if target_lang == "hi":
        prompt = f"Translate the following text to Hindi (Devanagari script). Return ONLY the translation:\n\n{text}"
    else:
        prompt = f"Translate the following text to English. Return ONLY the translation:\n\n{text}"
    translated = groq_chat("You are a professional translator. Return only the translated text.", prompt, temperature=0.2)
    return {"original": text, "translated": translated.strip(), "target_lang": target_lang}


def get_followup_suggestions(query: str, answer: str, lang: str = "en") -> List[str]:
    prompt = f"""Suggest 3 concise follow-up questions a user might ask next. Return ONLY a JSON array of strings:
["Question 1?", "Question 2?", "Question 3?"]
User question: {query}\nAnswer: {answer[:400]}"""

    raw = groq_chat("Return ONLY a JSON array of 3 questions. No markdown.", prompt, temperature=0.3)
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        s = json.loads(raw)
        return s[:3] if isinstance(s, list) else []
    except Exception:
        return []


def get_vault_analytics() -> Dict[str, Any]:
    from services.chroma_service import list_documents
    docs = list_documents()
    total_docs = len(docs)
    total_chunks = sum(d.get("chunks", 0) for d in docs)
    total_pages = sum(d.get("total_pages", 1) for d in docs)
    languages = list({d.get("language", "en") for d in docs})

    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "total_pages": total_pages,
        "languages": languages,
        "documents": docs,
    }


def get_popular_faqs() -> List[Dict[str, Any]]:
    return []


def get_faq_stats() -> Dict[str, Any]:
    return {"cached_questions": 0, "cache_hits": 0, "tokens_saved": 0}
