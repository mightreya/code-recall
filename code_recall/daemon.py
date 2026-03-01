"""Memory daemon: persistent HTTP server for fact extraction and recall.

Endpoints:
  POST /search  — JSON {query, collection, user_id, limit, expand} → JSON results
                   plain text body → formatted memory-context (Claude Code backward compat)
  POST /add     — text body (Claude Code default) or JSON {text, domain, collection, user_id}
  GET  /health  — returns 200
"""

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import httpx
from google.genai.errors import APIError

from code_recall._mem0 import build_memory
from code_recall.extract import (
    extract_facts,
    extract_workflow_state,
    hybrid_search,
    parse_timestamp,
    store_facts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 7377
SEARCH_LIMIT = 5
DEFAULT_COLLECTION = "mem0_dev"
DEFAULT_USER_ID = "developer"
DEFAULT_DOMAIN = "developer"

# Lazy-initialized mem0 Memory instances keyed by collection name
_memories: dict = {}


def _get_memory(collection: str):
    """Get or create a mem0 Memory instance for a collection."""
    if collection not in _memories:
        logger.info("Loading Mem0 for collection %s", collection)
        _memories[collection] = build_memory(collection)
    return _memories[collection]


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond(200, "ok")
        else:
            self._respond(404, "not found")

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()

        if self.path == "/search":
            self._handle_search(body)
        elif self.path == "/add":
            self._handle_add(body)
        elif self.path == "/capture-state":
            self._handle_capture_state(body)
        else:
            self._respond(404, "not found")

    def _handle_search(self, body: str) -> None:
        params = _parse_search_body(body)
        if not params["query"]:
            self._respond(200, "")
            return

        points = hybrid_search(
            collection=params["collection"],
            query=params["query"],
            user_id=params["user_id"],
            limit=params["limit"],
            expand=params["expand"],
        )

        # JSON request → JSON response; plain text → formatted memory-context
        if params["json_request"]:
            self._respond_json(200, points)
        else:
            formatted = _format_memory_context(points)
            self._respond(200, formatted)

    def _handle_add(self, body: str) -> None:
        if not body.strip():
            self._respond(204, "")
            return

        # JSON body = multi-domain request from OpenClaw plugins
        # Plain text body = Claude Code capture hook (backward compatible)
        params = _parse_add_body(body)
        if not params["text"]:
            self._respond(204, "")
            return

        Thread(target=_add_memory, args=(params,), daemon=True).start()
        self._respond(204, "")

    def _handle_capture_state(self, body: str) -> None:
        try:
            data = json.loads(body)
            text = data.get("text", "")
            workspace = data.get("workspace", "")
        except (json.JSONDecodeError, AttributeError):
            self._respond(400, "invalid json")
            return

        if not text or not workspace:
            self._respond(400, "text and workspace required")
            return

        Thread(target=_capture_workflow_state, args=(text, workspace), daemon=True).start()
        self._respond(204, "")

    def _respond(self, code: int, body: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        if body:
            self.wfile.write(body.encode())

    def _respond_json(self, code: int, data: list | dict) -> None:
        payload = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, /, *args: object) -> None:
        """Suppress default request logging."""


def _parse_search_body(body: str) -> dict:
    """Parse /search request body — JSON with params or plain text for backward compat."""
    try:
        data = json.loads(body)
        if isinstance(data, dict) and "query" in data:
            return {
                "query": data["query"],
                "collection": data.get("collection", DEFAULT_COLLECTION),
                "user_id": data.get("user_id", DEFAULT_USER_ID),
                "limit": data.get("limit", SEARCH_LIMIT),
                "expand": data.get("expand", False),
                "json_request": True,
            }
    except (json.JSONDecodeError, KeyError):
        pass
    return {
        "query": body.strip(),
        "collection": DEFAULT_COLLECTION,
        "user_id": DEFAULT_USER_ID,
        "limit": SEARCH_LIMIT,
        "expand": False,
        "json_request": False,
    }


def _format_memory_context(points: list[dict]) -> str:
    """Format hybrid search results as <memory-context> text for Claude Code hooks."""
    if not points:
        return ""
    lines = ["<memory-context>", "Relevant memories from previous sessions:"]
    for point in points:
        payload = point.get("payload", {})
        text = payload.get("data", "")
        if not text:
            continue
        raw_date = payload.get("sourced_at") or payload.get("updated_at") or payload.get("created_at") or ""
        timestamp = raw_date[:16].replace("T", " ") if len(raw_date) >= 16 else raw_date[:10]
        source = _format_source(payload.get("source", ""))
        labels = (timestamp, payload.get("project", ""), payload.get("category", ""), source)
        prefix = " ".join(f"[{label}]" for label in labels if label)
        lines.append(f"- {prefix} {text}" if prefix else f"- {text}")
    lines.append("</memory-context>")
    return "\n".join(lines)


def _format_source(source: str) -> str:
    """Convert source filename like 'session-05-sep-24-2025.md' to readable label."""
    if not source:
        return ""
    match = re.match(r"session-(\d+)", source)
    if not match:
        return ""
    session_number = int(match.group(1))
    return f"Therapy session #{session_number}"


def _parse_add_body(body: str) -> dict:
    """Parse /add request body — JSON with params or plain text for backward compat."""
    try:
        data = json.loads(body)
        if isinstance(data, dict) and "text" in data:
            return {
                "text": data["text"],
                "domain": data.get("domain", DEFAULT_DOMAIN),
                "collection": data.get("collection", DEFAULT_COLLECTION),
                "user_id": data.get("user_id", DEFAULT_USER_ID),
                "sourced_at": data.get("sourced_at", ""),
                "project": data.get("project", ""),
            }
    except (json.JSONDecodeError, KeyError):
        pass
    return {
        "text": body,
        "domain": DEFAULT_DOMAIN,
        "collection": DEFAULT_COLLECTION,
        "user_id": DEFAULT_USER_ID,
    }


def _add_memory(params: dict) -> None:
    """Extract structured facts and store in the appropriate collection."""
    try:
        memory = _get_memory(params["collection"])
        reference_date = parse_timestamp(params.get("sourced_at", ""))
        facts = extract_facts(params["text"], domain=params["domain"], reference_date=reference_date)
        sourced_at = params.get("sourced_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        metadata = {"sourced_at": sourced_at}
        if params.get("project"):
            metadata["project"] = params["project"]
        store_facts(memory, facts, params["user_id"], extra_metadata=metadata, collection_name=params["collection"])
        logger.info(
            "Extracted %d facts for %s (%d chars)",
            len(facts),
            params["collection"],
            len(params["text"]),
        )
    except Exception:
        logger.exception("Failed to capture memory for %s", params.get("collection", "unknown"))


def _capture_workflow_state(text: str, workspace: str) -> None:
    """Extract workflow state via Gemini and write to WORKFLOW_STATE.md atomically."""
    try:
        state = extract_workflow_state(text)
    except (APIError, httpx.HTTPError):
        logger.exception("Gemini extraction failed for %s", workspace)
        return

    if not state or state.strip() == "No active workflow.":
        logger.info("No active workflow detected, skipping state write")
        return

    target = Path(workspace) / "WORKFLOW_STATE.md"
    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(dir=workspace, suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            handle.write(state)
        os.rename(tmp_path, target)
    except OSError:
        logger.exception("Failed to write workflow state to %s", target)
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return

    logger.info("Wrote workflow state to %s (%d chars)", target, len(state))


def main() -> None:
    # Pre-load the default Claude Code collection
    _get_memory(DEFAULT_COLLECTION)
    logger.info("Starting server on %s:%d", HOST, PORT)

    server = HTTPServer((HOST, PORT), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.shutdown()
