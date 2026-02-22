"""Structured fact extraction and storage using Gemini with response_schema.

Shared by daemon.py, reingest.py, and ingest.py.
- extract_facts() calls Gemini directly for structured JSON output
- store_facts() writes extracted facts to Qdrant via mem0 with infer=False
"""

import json
import logging
import os

from google import genai
from google.genai import types

from code_recall.prompts import WORKFLOW_STATE_PROMPT, build_prompt

logger = logging.getLogger(__name__)

_MODEL = "gemini-3-flash-preview"
_MIN_SPECIFICITY = 3

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
                },
                "required": ["content", "category", "temporal_scope", "specificity_score"],
            },
        },
    },
    "required": ["facts"],
}

_MAX_RETRIES = 1
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazy-init Gemini client."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


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


def extract_facts(text: str, domain: str) -> list[dict]:
    """Extract structured facts from text using Gemini with guaranteed JSON schema.

    Returns list of dicts with keys: content, category, temporal_scope,
    specificity_score, source_type. Facts with specificity_score < 3 or
    temporal_scope == "ephemeral" are filtered out.
    """
    if not text or len(text.strip()) < 20:
        return []

    prompt = build_prompt(domain)
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
        if fact.get("specificity_score", 0) >= _MIN_SPECIFICITY and fact.get("temporal_scope") != "ephemeral"
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
    result = response.candidates[0].content.parts[0].text if response.candidates else ""
    return result.strip()


def store_facts(memory, facts: list[dict], user_id: str, extra_metadata: dict | None = None) -> int:
    """Store extracted facts in Qdrant via mem0 with infer=False. Returns count stored."""
    stored = 0
    for fact in facts:
        metadata = {
            "category": fact["category"],
            "temporal_scope": fact["temporal_scope"],
            "specificity_score": fact["specificity_score"],
            "source_type": fact.get("source_type", ""),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        memory.add(
            [{"role": "user", "content": fact["content"]}],
            user_id=user_id,
            infer=False,
            metadata=metadata,
        )
        stored += 1
    return stored
