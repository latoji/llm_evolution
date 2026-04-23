#!/usr/bin/env bash
# scripts/launch.sh — production launcher.
# Starts only the FastAPI backend (which also serves the built React frontend).
# No npm / Vite dev server needed.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/logs/backend.log"
PID_FILE="$ROOT/logs/backend.pid"

mkdir -p "$ROOT/logs"

# If already running, just open the browser.
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "LLM Evolution already running (PID $(cat "$PID_FILE")). Opening browser…"
    xdg-open http://localhost:8000 2>/dev/null || true
    exit 0
fi

# Activate venv
if [ -f "$ROOT/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$ROOT/venv/bin/activate"
fi

echo "Starting LLM Evolution backend…"
cd "$ROOT"
nohup python -m uvicorn api.main:app --port 8000 \
    > "$LOG" 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" > "$PID_FILE"
echo "Backend started (PID $BACKEND_PID). Log: $LOG"

# Wait for backend to be ready, then open browser
for i in $(seq 1 20); do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

xdg-open http://localhost:8000 2>/dev/null || \
    open http://localhost:8000 2>/dev/null || \
    echo "Open http://localhost:8000 in your browser."
