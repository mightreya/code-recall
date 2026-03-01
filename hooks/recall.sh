#!/bin/bash
# UserPromptSubmit hook: hybrid search via memory daemon, format as memory-context.
# Fail-open: if daemon is down, outputs nothing and exits 0.
DIR="$(dirname "$0")"
PROMPT=$(jq -r '.prompt // empty')
[ -z "$PROMPT" ] && exit 0
BODY=$(jq -nc --arg q "$PROMPT" '{query: $q, collection: "mem0_dev", user_id: "developer", limit: 5}')
curl -s --max-time 3 -H "Content-Type: application/json" -d "$BODY" http://127.0.0.1:7377/search 2>/dev/null \
  | jq -rf "$DIR/format-memories.jq"
exit 0
