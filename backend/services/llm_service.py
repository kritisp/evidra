"""
EVIDRA LLM & Embeddings Service
- LLM Generation: Groq API with round-robin key rotation and model fallback
- Embeddings: Google Gemini (models/gemini-embedding-001, 3072 dimensions)
"""

import sys
import time
import threading
from typing import Optional, List
import groq as groq_sdk
from google import genai
import config

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# 1. GROQ LLM SERVICE (Generation)
# ══════════════════════════════════════════════════════════════════════════════
_lock = threading.Lock()
_key_index = 0
_blocked_keys: set[str] = set()


def _next_groq_key(skip: Optional[set] = None) -> Optional[str]:
    """Return next available Groq key in round-robin order."""
    global _key_index
    skip = (skip or set()) | _blocked_keys
    available = [k for k in config.GROQ_API_KEYS if k not in skip]
    if not available:
        return None
    with _lock:
        key = available[_key_index % len(available)]
        _key_index += 1
    return key


def _classify_error(err: str) -> str:
    e = err.lower()
    if "401" in e or "invalid api key" in e or "authentication" in e:
        return "invalid_key"
    if "429" in e or "rate_limit" in e or "rate limit" in e:
        return "ratelimit"
    if "quota" in e or "exhausted" in e:
        return "quota"
    if "503" in e or "overloaded" in e or "unavailable" in e:
        return "overloaded"
    return "other"


def groq_chat(
    system_prompt: str,
    user_message: str,
    history: Optional[list] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    _model: Optional[str] = None,
    _is_fallback: bool = False,
) -> str:
    """
    Call Groq LLM with key rotation and graceful fallback.
    Low default temperature (0.2) to enforce strict grounding and minimize hallucinations.
    """
    if not config.GROQ_API_KEYS:
        raise RuntimeError("No Groq API keys configured. Set GROQ_API_KEY_1 in backend/.env")

    model = _model or config.GROQ_CHAT_MODEL
    tried_keys: set[str] = set()
    last_err = None
    max_attempts = config.RETRY_ATTEMPTS * max(len(config.GROQ_API_KEYS), 1)

    for attempt in range(max_attempts):
        key = _next_groq_key(skip=tried_keys)
        if key is None:
            if not _is_fallback and config.GROQ_FALLBACK_MODEL:
                print(f"[FALLBACK] Switching from {model} to fallback model {config.GROQ_FALLBACK_MODEL}...")
                return groq_chat(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    history=history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    _model=config.GROQ_FALLBACK_MODEL,
                    _is_fallback=True,
                )
            break

        try:
            client = groq_sdk.Groq(api_key=key)

            messages = [{"role": "system", "content": system_prompt}]
            for turn in (history or []):
                messages.append({
                    "role": "user" if turn.get("role") in ("user", "human") else "assistant",
                    "content": turn.get("content", ""),
                })
            messages.append({"role": "user", "content": user_message})

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return content if content is not None else ""

        except Exception as e:
            last_err = e
            kind = _classify_error(str(e))

            if kind == "invalid_key":
                print(f"[AUTH] Groq key invalid (suffix ...{key[-6:]})")
                _blocked_keys.add(key)
                tried_keys.add(key)
                continue

            if kind in ("ratelimit", "quota"):
                print(f"[RATE-LIMIT] Groq rate limit hit on {model} (attempt {attempt + 1}). Backing off...")
                time.sleep(3.0)
                if len(config.GROQ_API_KEYS) > 1:
                    tried_keys.add(key)
                continue

            if kind == "overloaded" and not _is_fallback:
                print(f"[OVERLOAD] Groq overloaded on {model}. Trying fallback...")
                return groq_chat(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    history=history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    _model=config.GROQ_FALLBACK_MODEL,
                    _is_fallback=True,
                )

            raise

    raise RuntimeError(
        f"All Groq attempts failed after {attempt + 1} tries (model: {model}). Last error: {last_err}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. GEMINI EMBEDDINGS SERVICE
# ══════════════════════════════════════════════════════════════════════════════
_gemini_client: Optional[genai.Client] = None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("No Gemini API key configured. Set GEMINI_API_KEY in backend/.env")
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def get_embedding(text: str) -> List[float]:
    """Generate embedding for a single text chunk or query."""
    client = _get_gemini_client()
    for attempt in range(config.RETRY_ATTEMPTS):
        try:
            res = client.models.embed_content(
                model=config.GEMINI_EMBED_MODEL,
                contents=text,
            )
            return list(res.embeddings[0].values)
        except Exception as e:
            if attempt < config.RETRY_ATTEMPTS - 1:
                time.sleep(config.RETRY_DELAY_SECONDS * (attempt + 1))
            else:
                raise RuntimeError(f"Gemini embedding failed: {e}") from e
    return []


def get_embeddings_batch(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    """Batch embed multiple text chunks."""
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for item in batch:
            vec = get_embedding(item)
            all_embeddings.append(vec)
    return all_embeddings


# Backwards compatibility aliases
gemini_embed = get_embeddings_batch
cohere_embed = get_embeddings_batch
gemini_embed_query = get_embedding
cohere_embed_query = get_embedding
gemini_chat = groq_chat
