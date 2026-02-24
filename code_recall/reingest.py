"""Wipe and rebuild a Qdrant collection from Claude Code transcript JSONL files.

Usage:
    code-recall-reingest                    # Re-ingest all transcripts
    code-recall-reingest --dry-run          # Extract facts without storing
    code-recall-reingest --wipe-only        # Just wipe, don't re-ingest
"""

import argparse
import json
import logging
import time
from pathlib import Path

import httpx
from google.genai.errors import ClientError, ServerError
from mem0 import Memory
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from code_recall._mem0 import QDRANT_URL, build_memory
from code_recall.daemon import DEFAULT_COLLECTION, DEFAULT_DOMAIN, DEFAULT_USER_ID
from code_recall.extract import extract_facts, parse_timestamp, store_facts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_TRANSCRIPTS_ROOT = Path.home() / ".claude" / "projects"
_EXTRACT_DELAY_SECONDS = 0.5

_GEMINI_RETRY = retry(
    retry=retry_if_exception_type((ClientError, ServerError)),
    wait=wait_exponential(multiplier=15, max=120),
    stop=stop_after_attempt(5),
    before_sleep=lambda state: logger.warning(
        "Gemini %s, retry %d in %.0fs",
        type(state.outcome.exception()).__name__,
        state.attempt_number,
        state.next_action.sleep,
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wipe and rebuild from Claude Code transcripts")
    parser.add_argument("--dry-run", action="store_true", help="Extract facts but don't store")
    parser.add_argument("--wipe-only", action="store_true", help="Just wipe collection, don't re-ingest")
    args = parser.parse_args()

    _wipe_collection(DEFAULT_COLLECTION)
    if args.wipe_only:
        return

    transcripts = _find_transcripts()
    logger.info("Found %d transcript files", len(transcripts))

    memory = None if args.dry_run else build_memory(DEFAULT_COLLECTION)

    total_facts = 0
    for index, path in enumerate(transcripts):
        count = _reingest_transcript(path, memory)
        total_facts += count
        logger.info(
            "  [%d/%d] %s → %d facts",
            index + 1, len(transcripts),
            path.parent.name[:20] + "/" + path.name[:12],
            count,
        )

    label = "(dry-run)" if args.dry_run else "stored"
    logger.info("Total: %d facts %s", total_facts, label)


def _reingest_transcript(path: Path, memory: Memory | None) -> int:
    """Extract and store facts from a single Claude Code transcript."""
    exchanges = _parse_transcript(path)
    if not exchanges:
        return 0

    count = 0
    for exchange_text, timestamp in exchanges:
        reference_date = parse_timestamp(timestamp)
        facts = _GEMINI_RETRY(extract_facts)(exchange_text, domain=DEFAULT_DOMAIN, reference_date=reference_date)
        if facts and memory:
            metadata = {"sourced_at": timestamp}
            project = _project_name(path)
            if project:
                metadata["project"] = project
            store_facts(memory, facts, DEFAULT_USER_ID, extra_metadata=metadata, collection_name=DEFAULT_COLLECTION)
        count += len(facts)
        if facts:
            time.sleep(_EXTRACT_DELAY_SECONDS)

    return count


def _wipe_collection(collection: str) -> None:
    """Delete all points from a Qdrant collection."""
    response = httpx.get(f"{QDRANT_URL}/collections/{collection}")
    if response.status_code != 200:
        logger.warning("[%s] Collection not found, skipping wipe", collection)
        return

    points_before = response.json()["result"]["points_count"]
    if points_before == 0:
        logger.info("[%s] Already empty", collection)
        return

    httpx.post(
        f"{QDRANT_URL}/collections/{collection}/points/delete",
        json={"filter": {}},
        timeout=60,
    )
    logger.info("[%s] Wiped %d points", collection, points_before)


def _find_transcripts() -> list[Path]:
    """Find all main Claude Code transcript JSONL files (skip subagents)."""
    paths = [path for path in _TRANSCRIPTS_ROOT.rglob("*.jsonl") if "subagents" not in path.parts]
    paths.sort(key=lambda path: path.stat().st_mtime)
    return paths


def _parse_transcript(path: Path) -> list[tuple[str, str]]:
    """Parse Claude Code transcript JSONL into (exchange_text, timestamp) pairs.

    Groups consecutive user+assistant messages into exchanges.
    """
    messages = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_type = entry.get("type", "")
            if entry_type not in ("user", "assistant"):
                continue
            timestamp = entry.get("timestamp", "")
            message = entry.get("message", {})
            content = message.get("content", "") if isinstance(message, dict) else ""
            text = _extract_text(content)
            if not text:
                continue
            messages.append((entry_type, text, timestamp))

    exchanges = []
    current_exchange = []
    current_timestamp = ""
    for role, text, timestamp in messages:
        if role == "user":
            if current_exchange:
                exchanges.append(("\n\n".join(current_exchange), current_timestamp))
            current_exchange = [f"User: {text}"]
            current_timestamp = timestamp
        elif role == "assistant" and current_exchange:
            current_exchange.append(f"Assistant: {text}")

    if current_exchange:
        exchanges.append(("\n\n".join(current_exchange), current_timestamp))

    return exchanges


def _extract_text(content: str | list) -> str:
    """Extract plain text from message content (string or content block array)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = (block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")
        joined = "\n".join(parts).strip()
        return joined
    return ""


def _project_name(path: Path) -> str:
    """Derive project name from transcript path."""
    relative = path.relative_to(_TRANSCRIPTS_ROOT)
    name = relative.parts[0] if relative.parts else ""
    return name
