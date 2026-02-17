#!/bin/bash
# Stop hook: capture conversation from transcript and send to memory daemon.
# Sends last 10 text messages for context-rich fact extraction.
# Tracks transcript growth to skip turns with no new content.
# Fail-open: if daemon is down or transcript is empty, exits 0 silently.
INPUT=$(cat)
ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')
[ "$ACTIVE" = "true" ] && exit 0

TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path // empty')
[ -z "$TRANSCRIPT" ] || [ ! -f "$TRANSCRIPT" ] && exit 0

# Skip if transcript hasn't grown since last capture (e.g. interrupted turn)
HASH=$(command -v md5sum >/dev/null && md5sum <<< "$TRANSCRIPT" || md5 -q -s "$TRANSCRIPT")
STATE="/tmp/cc-mem-$(echo "$HASH" | cut -c1-12)"
CAPTURED=$(cat "$STATE" 2>/dev/null || echo 0)
TOTAL=$(wc -l < "$TRANSCRIPT")
[ "$TOTAL" -le "$CAPTURED" ] && exit 0
echo "$TOTAL" > "$STATE"

# Derive project name from session working directory
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
PROJECT=$(basename -s .git "$(git -C "$CWD" remote get-url origin 2>/dev/null)" 2>/dev/null || basename "$CWD")

# Extract last 10 user/assistant text messages from the JSONL transcript.
# Context window ensures Gemini understands references; mem0 deduplicates overlap.
EXCHANGE=$(tail -2000 "$TRANSCRIPT" | jq -rs '
  def extract_text:
    if .message.content | type == "string" then .message.content
    elif .message.content | type == "array" then
      [.message.content[] | select(.type == "text") | .text] | join("\n")
    else "" end;
  def has_text:
    if .message.content | type == "string" then (.message.content | length > 0)
    elif .message.content | type == "array" then
      ([.message.content[] | select(.type == "text")] | length > 0)
    else false end;
  [.[] | select((.type == "user" or .type == "assistant") and has_text)] |
  .[-10:] |
  map(
    (if .type == "user" then "User" else "Assistant" end) as $label |
    "\($label): \(extract_text)"
  ) |
  if length > 0 then join("\n\n") else "" end
' 2>/dev/null)

[ -z "$EXCHANGE" ] && exit 0

# Send as JSON with project metadata, truncate text to 20KB
BODY=$(jq -nc --arg text "${EXCHANGE:0:20000}" --arg project "$PROJECT" \
  '{text: $text, project: $project}')
echo "$BODY" | curl -s --max-time 5 -H "Content-Type: application/json" -d @- http://127.0.0.1:7377/add >/dev/null 2>&1 &
exit 0
