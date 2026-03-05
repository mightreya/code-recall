# code-recall

Persistent memory for [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Remembers facts from your conversations and recalls them automatically in future sessions.

After each session, relevant facts are extracted and stored. Before each prompt, matching memories are injected as context — so Claude already knows your project structure, tooling preferences, debugging patterns, and workflow decisions. You stop repeating yourself.

## Quick start

### Prerequisites

- [uv](https://docs.astral.sh/uv/), [jq](https://jqlang.github.io/jq/), `curl`, `git`
- [Docker](https://docs.docker.com/get-docker/) (for Qdrant)
- [Ollama](https://ollama.com/)
- A free [Gemini API key](https://aistudio.google.com/apikey)

### 1. Start infrastructure

```bash
# Qdrant vector database
docker run -d --name qdrant --restart always \
  -p 6333:6333 -p 6334:6334 \
  -v ~/.qdrant/storage:/qdrant/storage qdrant/qdrant

# Ollama embedding model
ollama pull bge-m3
```

### 2. Install

```bash
git clone https://github.com/mightreya/code-recall.git ~/code-recall
cd ~/code-recall && uv sync
chmod +x hooks/*.sh
```

### 3. Start the daemon

**Option A: systemd (Linux)**

```bash
mkdir -p ~/.config/systemd/user
cp ~/code-recall/code-recall.service ~/.config/systemd/user/
sed -i "s/YOUR_KEY_HERE/$GEMINI_API_KEY/" ~/.config/systemd/user/code-recall.service
systemctl --user daemon-reload
systemctl --user enable --now code-recall
```

**Option B: launchd (macOS)**

```bash
export GEMINI_API_KEY="your-key-here"

cat > ~/Library/LaunchAgents/com.code-recall.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.code-recall</string>
  <key>ProgramArguments</key>
  <array>
    <string>$HOME/code-recall/.venv/bin/code-recall-daemon</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>GEMINI_API_KEY</key>
    <string>$GEMINI_API_KEY</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/code-recall.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/code-recall.log</string>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.code-recall.plist
```

**Option C: run manually**

```bash
GEMINI_API_KEY="your-key-here" ~/code-recall/.venv/bin/code-recall-daemon
```

### 4. Register hooks

Add to `~/.claude/settings.json` (merge with existing settings if any):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/code-recall/hooks/recall.sh"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/code-recall/hooks/capture.sh"
          }
        ]
      }
    ]
  }
}
```

### 5. Verify

```bash
# Check daemon health
curl -s http://127.0.0.1:7377/health
# → ok

# Store a test fact
curl -s -X POST http://127.0.0.1:7377/add \
  -H 'Content-Type: application/json' \
  -d '{"text": "User: We use pytest with coverage and ruff for linting."}'

# Wait for extraction, then search
sleep 5
curl -s -X POST http://127.0.0.1:7377/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "linting setup"}'
# → JSON array of matching memories
```

Start a Claude Code session. You should see memories injected automatically.

## Reingest existing sessions

If you have existing Claude Code transcript history (in `~/.claude/projects/`), you can bulk-ingest all past sessions:

```bash
# Wipe the collection and re-ingest everything
cd ~/code-recall && uv run code-recall-reingest

# Preview what would be extracted without storing
cd ~/code-recall && uv run code-recall-reingest --dry-run

# Wipe the collection without re-ingesting (fresh start)
cd ~/code-recall && uv run code-recall-reingest --wipe-only
```

Reingest discovers all `.jsonl` transcript files under `~/.claude/projects/`, extracts facts via Gemini with original timestamps preserved, and stores them in Qdrant with deduplication. Rate-limited to ~2 requests/second.

## Nightly consolidation

Over time, near-duplicate facts accumulate. The consolidation job deduplicates by vector similarity (0.80 threshold, keeps newer) and prunes expired situational facts.

```bash
# Run manually
cd ~/code-recall && uv run code-recall-consolidate

# Install as nightly systemd timer (Linux)
cp ~/code-recall/code-recall-consolidate.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now code-recall-consolidate.timer
```

## How it works

```
User prompt ──→ recall.sh ──→ POST /search ──→ Qdrant (dense + BM25 hybrid)
  (jq + curl)    ~200ms         daemon
  injects <memory-context>

Session end ──→ capture.sh ──→ POST /add ──→ Gemini (extraction)
  (jq + curl)     ~50ms          daemon     → Ollama BGE-M3 (embedding)
  fire-and-forget                           → Qdrant (storage)
                                            → Mem0 (deduplication)
```

- **Recall** runs before every prompt. Performs hybrid search (semantic embeddings + BM25 keyword matching) via Reciprocal Rank Fusion and injects matching memories.
- **Capture** runs after every session. Extracts the last 10 messages from the transcript, sends them to Gemini for structured fact extraction with quality scoring. Low-specificity and ephemeral facts are filtered out.
- **Fail-open**: both hooks exit 0 on any error. A dead daemon never breaks Claude Code.

## Custom domains

The built-in `developer` domain extracts codebase structure, tooling, debugging patterns, and workflow preferences. To add custom domains, create a YAML file:

```yaml
my_domain: |
  You are a fact extractor for ...

  ## CATEGORIES TO EXTRACT
  - Category one
  - Category two
```

Set `MEMORY_DOMAINS_FILE` in the daemon environment:

```bash
Environment=MEMORY_DOMAINS_FILE=/path/to/domains.yaml
```

Then use it via the `/add` endpoint with `"domain": "my_domain"`.

## Troubleshooting

```bash
# Daemon logs (Linux)
journalctl --user -u code-recall -f

# Daemon logs (macOS)
tail -f /tmp/code-recall.log

# Check Qdrant has data
curl -s http://localhost:6333/collections/mem0_dev | jq .result.points_count

# Restart daemon (Linux / macOS)
systemctl --user restart code-recall
launchctl kickstart -k gui/$(id -u)/com.code-recall
```

## License

[MIT](LICENSE)
