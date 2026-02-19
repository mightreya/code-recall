"""Fact extraction prompts for Mem0.

Architecture:
  - build_prompt(domain) → extraction instructions only (no output format)
  - build_mem0_prompt(domain) → wraps build_prompt() + mem0 {"facts": []} format suffix
  - Set MEMORY_DOMAINS_FILE env var to load additional domains from a YAML file
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Shared preamble and rules (appended to every domain prompt)

_SHARED_RULES = """\

## EXTRACTION RULES
1. Every fact MUST contain concrete specifics: names, numbers, versions, dates, or exact preferences. \
Never use "recently," "something," "various," or "some."
2. Every fact MUST be self-contained — fully understandable without seeing the conversation.
3. Extract PATTERNS and KNOWLEDGE, not individual events or activities.
4. Extract ONLY from USER messages. Assistant messages provide context but NEVER extract facts \
from them — the assistant can misinterpret, hallucinate, or state incorrect information.
5. If nothing is memory-worthy, return an empty list of facts.

## DO NOT EXTRACT
- Activity logs: "merged a branch," "sent an email," "ran a test," "is debugging"
- Temporary states: "is currently testing X," "is frustrated," "is working on Y right now"
- Vague summaries: "discussed work topics," "shared preferences," "talked about technical stuff"
- Common knowledge: "uses a computer," "writes code," "has a job"
- Repetitions of already-known information

## TEMPORAL CLASSIFICATION — apply before storing:
- PERMANENT: Unchanging (name, birthday, native language, identity facts)
- STABLE: True for months/years, may evolve (job, tools, core preferences, relationships)
- SITUATIONAL: True for weeks (current project phase, upcoming events with dates)
- EPHEMERAL: True for hours/days → DO NOT EXTRACT"""


def _date_line() -> str:
    return f"\n\nToday's date is {datetime.now().strftime('%Y-%m-%d')}."


# Domain-specific instruction blocks

_DEVELOPER_INSTRUCTIONS = """\
You are a developer workflow knowledge curator. Extract facts about the developer's codebase, tooling, \
workflow preferences, debugging patterns, and project architecture — information that makes future \
coding assistance more precise and contextual.

## CATEGORIES TO EXTRACT
- Codebase structure (monorepo vs multi-repo, key directories, module organization)
- Project-level configurations (language versions, package managers, build tools, linters)
- Workflow preferences ("always run tests before committing," "prefers small PRs")
- Debugging patterns learned ("error X in this project is usually caused by Y")
- Tool configurations (editor settings, terminal setup, shell aliases)
- Code style preferences beyond linting (naming conventions, abstraction preferences)
- Deployment pipeline specifics (staging → production flow, CI/CD tools)
- Dependency management choices (pinned versions, preferred libraries)
- Testing philosophy and tools (unit vs integration priorities, test frameworks)
- Known project-specific gotchas ("the auth module uses a custom token format, not standard JWT")

## DO NOT EXTRACT (in addition to general rules)
- Individual debugging steps from a single session
- File contents or code snippets — store knowledge ABOUT the code, not the code itself
- Error messages without generalizable lessons
- Temporary workarounds: "commenting out line 42 for now"

## EXAMPLES

Conversation: "Oh yeah, in this project we use pnpm workspaces with turborepo, and the API is in \
packages/api while the frontend is packages/web"
GOOD: "Project uses pnpm workspaces with Turborepo monorepo. API at packages/api, frontend at packages/web"
REJECTED: "Uses a monorepo" — loses the specific tools and structure

Conversation: "Every time I see ECONNREFUSED on port 5432, it's because the Docker Postgres container \
isn't running"
GOOD: "ECONNREFUSED on port 5432 means Docker PostgreSQL container is not running"
REJECTED: "Had a database error" — loses the debugging pattern"""


_DOMAIN_INSTRUCTIONS: dict[str, str] = {
    "developer": _DEVELOPER_INSTRUCTIONS,
}

# Load additional domains from YAML file

_domains_file = os.environ.get("MEMORY_DOMAINS_FILE")
if _domains_file:
    _domains_path = Path(_domains_file)
    if _domains_path.is_file():
        _extra = yaml.safe_load(_domains_path.read_text())
        _DOMAIN_INSTRUCTIONS.update(_extra)
        logger.info("Loaded %d domain(s) from %s", len(_extra), _domains_path.name)

# Mem0 output format suffix

_MEM0_FORMAT_SUFFIX = """

Return the facts in JSON format with a key "facts" and a list of strings:
{"facts": ["Self-contained fact with specific details", "Another specific fact"]}

If nothing is memory-worthy, return: {"facts": []}

Remember:
- Make sure to return the response in JSON with a key as "facts" and corresponding value will be a \
list of strings.
- You should detect the language of the user input and record the facts in the same language.
"""

# Workflow state extraction prompt (domain-agnostic)

WORKFLOW_STATE_PROMPT = """\
Extract the current workflow state from this conversation.
Return a markdown document with these sections:

# Workflow State
> Updated: {timestamp}

## Active Goal
What the user is trying to accomplish. Status: IN_PROGRESS / BLOCKED / COMPLETED

## Progress
- [x] Completed steps
- [ ] **→ Current step** ← NEXT
- [ ] Remaining steps

## Context
- **Key files/dirs**: paths mentioned
- **Last action**: what was just done
- **Next action**: what should happen next
- **Blockers**: what's waiting on user or external input

## Decisions
Key decisions made during this conversation.

If no active workflow (casual conversation), return exactly: "No active workflow."
Keep output under 500 tokens. Be specific — file paths, item counts, exact status.\
"""


def build_prompt(domain: str) -> str:
    """Return extraction instructions for a domain, without output format.

    Used by extract.py where Gemini response_schema handles the output format.
    """
    if domain not in _DOMAIN_INSTRUCTIONS:
        raise ValueError(f"Unknown domain: {domain!r}. Valid: {', '.join(_DOMAIN_INSTRUCTIONS)}")
    instructions = _DOMAIN_INSTRUCTIONS[domain]
    return instructions + _SHARED_RULES + _date_line()


def build_mem0_prompt(domain: str) -> str:
    """Return extraction instructions with mem0 {"facts": []} output format.

    Used by OpenClaw bot configs (customPrompt) where mem0 parses the output.
    """
    return build_prompt(domain) + _MEM0_FORMAT_SUFFIX
