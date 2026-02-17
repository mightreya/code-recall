#!/bin/bash
# UserPromptSubmit hook: query memory daemon for relevant memories.
# Fail-open: if daemon is down, outputs nothing and exits 0.
PROMPT=$(jq -r '.prompt // empty')
[ -z "$PROMPT" ] && exit 0
curl -s --max-time 2 -d "$PROMPT" http://127.0.0.1:7377/search 2>/dev/null
exit 0
