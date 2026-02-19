#!/usr/bin/env python3
"""Memory daemon: persistent HTTP server for fact extraction and recall.

Endpoints:
  POST /search  — body is the query text, returns memory-context plain text
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

from memory._mem0 import build_memory
from memory.extract import extract_facts, extract_workflow_state, store_facts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 7377
SEARCH_LIMIT = 5
_DEFAULT_COLLECTION = "mem0_claude_code"
_DEFAULT_USER_ID = "claude_code"
_DEFAULT_DOMAIN = "developer"

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

    def _handle_search(self, query: str) -> None:
        if not query.strip():
            self._respond(200, "")
            return

        memory = _get_memory(_DEFAULT_COLLECTION)
        results = memory.search(query, user_id=_DEFAULT_USER_ID, limit=SEARCH_LIMIT)
        memories = results.get("results", results) if isinstance(results, dict) else results
        if not memories:
            self._respond(200, "")
            return

        lines = ["<memory-context>", "Relevant memories from previous sessions:"]
        for entry in memories:
            text = entry.get("memory", "") if isinstance(entry, dict) else str(entry)
            if not text:
                continue
            metadata = entry.get("metadata", {}) or {}
            raw_date = metadata.get("sourced_at") or entry.get("updated_at") or entry.get("created_at") or ""
            timestamp = raw_date[:16].replace("T", " ") if len(raw_date) >= 16 else raw_date[:10]
            source = _format_source(metadata.get("source", ""))
            labels = [timestamp, metadata.get("project", ""), metadata.get("category", ""), source]
            prefix = " ".join(f"[{label}]" for label in labels if label)
            lines.append(f"- {prefix} {text}" if prefix else f"- {text}")
        lines.append("</memory-context>")

        self._respond(200, "\n".join(lines))

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

    def log_message(self, format: str, /, *args: object) -> None:
        """Suppress default request logging."""


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
                "domain": data.get("domain", _DEFAULT_DOMAIN),
                "collection": data.get("collection", _DEFAULT_COLLECTION),
                "user_id": data.get("user_id", _DEFAULT_USER_ID),
                "sourced_at": data.get("sourced_at", ""),
                "project": data.get("project", ""),
            }
    except (json.JSONDecodeError, KeyError):
        pass
    return {
        "text": body,
        "domain": _DEFAULT_DOMAIN,
        "collection": _DEFAULT_COLLECTION,
        "user_id": _DEFAULT_USER_ID,
    }


def _add_memory(params: dict) -> None:
    """Extract structured facts and store in the appropriate collection."""
    try:
        memory = _get_memory(params["collection"])
        facts = extract_facts(params["text"], domain=params["domain"])
        sourced_at = params.get("sourced_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        metadata = {"sourced_at": sourced_at}
        if params.get("project"):
            metadata["project"] = params["project"]
        store_facts(memory, facts, params["user_id"], extra_metadata=metadata)
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
    _get_memory(_DEFAULT_COLLECTION)
    logger.info("Starting server on %s:%d", HOST, PORT)

    server = HTTPServer((HOST, PORT), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
