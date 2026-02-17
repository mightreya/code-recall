# claude-memory

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
git clone https://github.com/mightreya/claude-memory.git ~/claude-memory
cd ~/claude-memory && uv sync
chmod +x hooks/*.sh
```

### 3. Start the daemon

**Option A: systemd (Linux)**

```bash
mkdir -p ~/.config/systemd/user
cp ~/claude-memory/claude-memory.service ~/.config/systemd/user/
sed -i "s/YOUR_KEY_HERE/$GEMINI_API_KEY/" ~/.config/systemd/user/claude-memory.service
systemctl --user daemon-reload
systemctl --user enable --now claude-memory
```

**Option B: launchd (macOS)**

```bash
export GEMINI_API_KEY="your-key-here"

cat > ~/Library/LaunchAgents/com.claude-memory.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.claude-memory</string>
  <key>ProgramArguments</key>
  <array>
    <string>$HOME/claude-memory/.venv/bin/python</string>
    <string>$HOME/claude-memory/daemon.py</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONPATH</key>
    <string>$HOME/claude-memory</string>
    <key>GEMINI_API_KEY</key>
    <string>$GEMINI_API_KEY</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/claude-memory.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/claude-memory.log</string>
</dict>
</plist>
EOF

launchctl load ~/Library/LaunchAgents/com.claude-memory.plist
```

**Option C: run manually**

```bash
GEMINI_API_KEY="your-key-here" PYTHONPATH=~/claude-memory \
  ~/claude-memory/.venv/bin/python ~/claude-memory/daemon.py
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
            "command": "$HOME/claude-memory/hooks/recall.sh"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/claude-memory/hooks/capture.sh"
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
curl -s -d "linting setup" http://127.0.0.1:7377/search
# → <memory-context> with the captured fact
```

Start a Claude Code session. You should see memories injected automatically.

## How it works

```
User prompt ──→ recall.sh ──→ POST /search ──→ Qdrant
  (jq + curl)    ~200ms         daemon
  injects <memory-context>

Session end ──→ capture.sh ──→ POST /add ──→ Gemini (extraction)
  (jq + curl)     ~50ms          daemon     → Ollama BGE-M3 (embedding)
  fire-and-forget                           → Qdrant (storage)
                                            → Mem0 (deduplication)
```

- **Recall** runs before every prompt. Searches Qdrant via BGE-M3 embeddings and injects matching memories.
- **Capture** runs after every response. Extracts the last 10 messages from the transcript, sends them to Gemini for structured fact extraction with quality scoring. Low-specificity and ephemeral facts are filtered out.
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
journalctl --user -u claude-memory -f

# Daemon logs (macOS)
tail -f /tmp/claude-memory.log

# Check Qdrant has data
curl -s http://localhost:6333/collections/mem0_dev | jq .result.points_count

# Restart daemon (Linux / macOS)
systemctl --user restart claude-memory
launchctl kickstart -k gui/$(id -u)/com.claude-memory
```

## License

[MIT](LICENSE)
