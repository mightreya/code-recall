"""Shared Mem0 initialization with Anthropic bug workarounds."""

import os

from mem0 import Memory
from mem0.llms.anthropic import AnthropicLLM
from mem0.memory.utils import remove_code_blocks

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# Class-level monkey-patch for AnthropicLLM (mem0ai v1.0.3)
#
# Bug 1: BaseLlmConfig defaults both temperature=0.1 and top_p=0.1;
#         Claude 4.x hard-rejects the combination with HTTP 400.
#         Setting top_p=None still fails (SDK sends null to API).
# Bug 2: response_format={"type": "json_object"} accepted but ignored;
#         Claude wraps JSON in code fences → JSONDecodeError → silent data loss.
#
# Upstream: PR #3732 open since Nov 2025, unmerged. No fix in v1.0.3.
# Research: .docs/mem0-anthropic-bugs-research.md


def _patched_generate(self, messages, response_format=None, tools=None, tool_choice="auto", **kwargs):
    system_message = ""
    filtered_messages = []
    for message in messages:
        if message["role"] == "system":
            system_message = message["content"]
        else:
            filtered_messages.append(message)

    # Bug 1: build params without top_p/top_k (Claude 4.x rejects them)
    params = {
        "model": self.config.model,
        "messages": filtered_messages,
        "system": system_message,
        "temperature": self.config.temperature,
        "max_tokens": self.config.max_tokens,
    }

    if tools:
        params["tools"] = tools
        params["tool_choice"] = tool_choice

    response = self.client.messages.create(**params)
    result = response.content[0].text

    # Bug 2: strip markdown code fences that Claude wraps around JSON
    if isinstance(result, str):
        result = remove_code_blocks(result)
    return result


AnthropicLLM.generate_response = _patched_generate


def build_memory(collection_name: str, extraction_prompt: str | None = None) -> Memory:
    """Initialize Mem0 with local Qdrant + Ollama BGE-M3 + Gemini."""
    config = {
        "embedder": {
            "provider": "ollama",
            "config": {"model": "bge-m3", "ollama_base_url": OLLAMA_URL},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": QDRANT_HOST,
                "port": QDRANT_PORT,
                "collection_name": collection_name,
                "embedding_model_dims": 1024,
            },
        },
        "llm": {
            "provider": "gemini",
            "config": {
                "model": "gemini-3-flash-preview",
                "api_key": os.environ["GEMINI_API_KEY"],
                "temperature": 0,
            },
        },
    }
    if extraction_prompt:
        config["custom_fact_extraction_prompt"] = extraction_prompt
    return Memory.from_config(config)
