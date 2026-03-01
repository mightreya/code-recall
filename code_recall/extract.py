"""Structured fact extraction, storage, and hybrid search using Gemini + Qdrant.

Shared by daemon.py, reingest.py, and ingest.py.
- extract_facts() calls Gemini directly for structured JSON output
- store_facts() writes extracted facts to Qdrant via mem0 with infer=False
- hybrid_search() does dense+sparse retrieval with RRF fusion
"""

import functools
import json
import logging
import os
from datetime import datetime

import httpx
from fastembed import SparseTextEmbedding
from google import genai
from google.genai import types

from code_recall._mem0 import OLLAMA_URL, QDRANT_URL
from code_recall.prompts import WORKFLOW_STATE_PROMPT, build_prompt

logger = logging.getLogger(__name__)

_MODEL = "gemini-3-flash-preview"
_MIN_SPECIFICITY = 3
_DEDUP_THRESHOLD = 0.85
_MIN_UNIQUE_WORDS = 15
_DEDUP_COSINE_WEIGHT = 0.7
_DEDUP_JACCARD_WEIGHT = 0.3
_PREFETCH_LIMIT = 20
_EXPAND_PROMPT = (
    "Rewrite this search query 3 different ways to capture different vocabulary. "
    "Resolve any pronouns using context. Return a JSON array of strings.\nQuery: {query}"
)

_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "facts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "content": {"type": "STRING"},
                    "category": {"type": "STRING"},
                    "temporal_scope": {
                        "type": "STRING",
                        "enum": ["permanent", "stable", "situational", "ephemeral"],
                    },
                    "specificity_score": {"type": "INTEGER"},
                    "source_type": {
                        "type": "STRING",
                        "enum": ["explicit_statement", "strong_inference", "weak_inference"],
                    },
                    "entities": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "valid_at": {"type": "STRING"},
                    "expires_at": {"type": "STRING"},
                },
                "required": ["content", "category", "temporal_scope", "specificity_score", "entities"],
            },
        },
    },
    "required": ["facts"],
}

_MAX_RETRIES = 1
_client: genai.Client | None = None


def hybrid_search(
    collection: str,
    query: str,
    user_id: str,
    limit: int = 7,
    expand: bool = False,
) -> list[dict]:
    """Dense+BM25 hybrid search with RRF fusion. Optionally expands queries via Gemini Flash."""
    queries = _expand_queries(query) if expand else (query,)

    user_filter = {
        "should": [
            {"key": "user_id", "match": {"value": user_id}},
            {"key": "userId", "match": {"value": user_id}},
        ],
    }

    prefetch = []
    for variant in queries:
        dense = _embed_text(variant)
        if dense:
            prefetch.append({"query": dense, "limit": _PREFETCH_LIMIT, "filter": user_filter})

        sparse = _sparse_encode(variant)
        if sparse:
            prefetch.append({"query": sparse, "using": "sparse", "limit": _PREFETCH_LIMIT, "filter": user_filter})

    if not prefetch:
        return []

    response = httpx.post(
        f"{QDRANT_URL}/collections/{collection}/points/query",
        json={"prefetch": prefetch, "query": {"fusion": "rrf"}, "limit": limit, "with_payload": True},
    )
    points = response.json().get("result", {}).get("points", [])
    return points


def extract_facts(text: str, domain: str, reference_date: datetime | None = None) -> list[dict]:
    """Extract structured facts from text using Gemini with guaranteed JSON schema.

    Pipeline: entropy pre-filter → Gemini extraction → specificity/temporal/entity post-filter.
    When reference_date is provided, the LLM sees that date instead of today —
    ensures valid_at reflects when conversations happened, not when reingest runs.
    """
    if not text or len(text.strip()) < 20:
        return []

    if _is_low_entropy(text):
        logger.debug("Skipping low-entropy text (%d chars, <%d unique words)", len(text), _MIN_UNIQUE_WORDS)
        return []

    prompt = build_prompt(domain, reference_date)
    client = _get_client()

    config = types.GenerateContentConfig(
        system_instruction=prompt,
        temperature=0,
        max_output_tokens=16384,
        response_mime_type="application/json",
        response_schema=_RESPONSE_SCHEMA,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    )

    contents = [types.Content(parts=[types.Part(text=f"Input:\n{text}")], role="user")]

    data = None
    for attempt in range(_MAX_RETRIES + 1):
        response = client.models.generate_content(model=_MODEL, contents=contents, config=config)
        data = _parse_response(response)
        if data is not None:
            break
        if attempt < _MAX_RETRIES:
            logger.info("Retrying extraction (attempt %d/%d)", attempt + 2, _MAX_RETRIES + 1)

    if data is None:
        logger.warning("Extraction failed after %d attempts, skipping", _MAX_RETRIES + 1)
        return []

    facts = data.get("facts", [])

    filtered = [
        fact
        for fact in facts
        if fact.get("specificity_score", 0) >= _MIN_SPECIFICITY
        and fact.get("temporal_scope") != "ephemeral"
        and len(fact.get("entities", ())) > 0
    ]

    logger.debug("Extracted %d facts (%d passed filter) from %d chars", len(facts), len(filtered), len(text))
    return filtered


def extract_workflow_state(text: str) -> str:
    """Extract structured workflow state from conversation text using Gemini.

    Returns raw markdown string suitable for WORKFLOW_STATE.md.
    Unlike extract_facts(), no JSON schema — Gemini returns free-form markdown.
    """
    if not text or len(text.strip()) < 50:
        return ""

    client = _get_client()
    config = types.GenerateContentConfig(
        system_instruction=WORKFLOW_STATE_PROMPT,
        temperature=0,
        max_output_tokens=2048,
    )
    contents = [types.Content(parts=[types.Part(text=text)], role="user")]
    response = client.models.generate_content(model=_MODEL, contents=contents, config=config)
    raw_text = response.candidates[0].content.parts[0].text if response.candidates else ""
    state = raw_text.strip()
    return state


def store_facts(
    memory,
    facts: list[dict],
    user_id: str,
    extra_metadata: dict | None = None,
    collection_name: str | None = None,
) -> int:
    """Store extracted facts in Qdrant via mem0 with infer=False. Returns count stored.

    When collection_name is provided, each fact is checked for near-duplicates
    before insertion (~35ms overhead per fact). After mem0 stores the dense vector,
    a BM25 sparse vector is added for hybrid retrieval. Both dedup and sparse are fail-open.
    """
    stored = 0
    skipped = 0
    for fact in facts:
        if collection_name and _is_duplicate(collection_name, fact["content"]):
            skipped += 1
            continue
        metadata = {
            "category": fact["category"],
            "temporal_scope": fact["temporal_scope"],
            "specificity_score": fact["specificity_score"],
            "source_type": fact.get("source_type", ""),
            "entities": ", ".join(fact.get("entities", ())),
        }
        valid_at = fact.get("valid_at")
        if valid_at:
            metadata["valid_at"] = valid_at
        expires_at = fact.get("expires_at")
        if expires_at:
            metadata["expires_at"] = expires_at
        if extra_metadata:
            metadata.update(extra_metadata)
        result = memory.add(
            [{"role": "user", "content": fact["content"]}],
            user_id=user_id,
            infer=False,
            metadata=metadata,
        )
        stored += 1
        if collection_name:
            _upsert_sparse_vector(collection_name, result, fact["content"])
    if skipped:
        logger.info("Dedup: skipped %d/%d duplicate facts", skipped, skipped + stored)
    return stored


def parse_timestamp(timestamp: str) -> datetime | None:
    """Parse an ISO-8601 timestamp string to datetime. Returns None on empty/invalid input."""
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return parsed
    except ValueError:
        return None


def _get_client() -> genai.Client:
    """Lazy-init Gemini client."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


@functools.cache
def _get_sparse_model() -> SparseTextEmbedding:
    """Lazy-init fastembed BM25 sparse text embedding model."""
    logger.info("Loading fastembed BM25 sparse model")
    return SparseTextEmbedding(model_name="Qdrant/bm25")


def _parse_response(response) -> dict | None:
    """Parse Gemini response JSON, returning None on failure."""
    text = response.candidates[0].content.parts[0].text if response.candidates else ""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Gemini returned invalid JSON (%d chars)", len(text))
        return None


def _expand_queries(query: str) -> tuple[str, ...]:
    """Generate query variants via Gemini Flash for multi-query expansion."""
    try:
        client = _get_client()
        config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=512,
            response_mime_type="application/json",
        )
        prompt = _EXPAND_PROMPT.format(query=query)
        contents = [types.Content(parts=[types.Part(text=prompt)], role="user")]
        response = client.models.generate_content(model=_MODEL, contents=contents, config=config)
        text = response.candidates[0].content.parts[0].text if response.candidates else ""
        variants = json.loads(text) if text else []
        if isinstance(variants, list):
            extras = tuple(variant for variant in variants if isinstance(variant, str))[:3]
            return (query, *extras)
    except (json.JSONDecodeError, IndexError, AttributeError):
        logger.debug("Query expansion failed, using original", exc_info=True)
    return (query,)


def _embed_text(text: str) -> list[float] | None:
    """Embed text via Ollama BGE-M3. Returns None on failure."""
    try:
        response = httpx.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": "bge-m3", "input": text},
        )
        embeddings = response.json().get("embeddings", [])
        return embeddings[0] if embeddings else None
    except (httpx.HTTPError, KeyError, IndexError):
        logger.debug("Embedding failed: %s", text[:80], exc_info=True)
        return None


def _sparse_encode(text: str) -> dict | None:
    """Encode text as BM25 sparse vector via fastembed. Returns {indices, values} or None."""
    try:
        model = _get_sparse_model()
        embedding = next(model.embed((text,)))
        return {"indices": embedding.indices.tolist(), "values": embedding.values.tolist()}
    except (StopIteration, RuntimeError):
        logger.debug("Sparse encoding failed: %s", text[:80], exc_info=True)
        return None


def _upsert_sparse_vector(collection: str, add_result: dict, text: str) -> None:
    """Add BM25 sparse vector to a just-stored point. Fail-open."""
    try:
        results = add_result.get("results", []) if isinstance(add_result, dict) else []
        for item in results:
            point_id = item.get("id")
            if not point_id or item.get("event") not in ("ADD", "UPDATE"):
                continue
            sparse = _sparse_encode(text)
            if not sparse:
                continue
            httpx.put(
                f"{QDRANT_URL}/collections/{collection}/points/vectors",
                json={"points": [{"id": point_id, "vector": {"sparse": sparse}}]},
            )
    except (httpx.HTTPError, KeyError, TypeError):
        logger.debug("Sparse vector upsert failed: %s", text[:80], exc_info=True)


def _is_low_entropy(text: str) -> bool:
    """Check if text has too few unique words to warrant a Gemini API call."""
    if len(text) < 100:
        return True
    words = {word.lower() for word in text.split() if len(word) > 2}
    return len(words) < _MIN_UNIQUE_WORDS


def _is_duplicate(collection: str, text: str) -> bool:
    """Hybrid dedup: weighted combination of cosine similarity and word overlap."""
    try:
        embedding = _embed_text(text)
        if not embedding:
            return False
        response = httpx.post(
            f"{QDRANT_URL}/collections/{collection}/points/search",
            json={"vector": embedding, "limit": 1, "with_payload": True},
        )
        results = response.json().get("result", [])
        if not results:
            return False
        cosine_score = results[0]["score"]
        existing_text = results[0].get("payload", {}).get("data", "")
        jaccard = _jaccard_word_overlap(text, existing_text) if existing_text else 0.0
        hybrid_score = _DEDUP_COSINE_WEIGHT * cosine_score + _DEDUP_JACCARD_WEIGHT * jaccard
        if hybrid_score >= _DEDUP_THRESHOLD:
            logger.debug(
                "Duplicate (hybrid=%.3f, cos=%.3f, jac=%.3f): %s", hybrid_score, cosine_score, jaccard, text[:80]
            )
            return True
    except (httpx.HTTPError, KeyError, IndexError):
        logger.debug("Dedup check failed for: %s", text[:80], exc_info=True)
    return False


def _jaccard_word_overlap(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity between two texts."""
    words_a = {word.lower() for word in text_a.split() if len(word) > 2}
    words_b = {word.lower() for word in text_b.split() if len(word) > 2}
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union
