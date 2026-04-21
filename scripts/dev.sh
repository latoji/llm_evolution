#!/usr/bin/env bash
# scripts/dev.sh — start backend + frontend for local development.
# Usage: bash scripts/dev.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Activate venv if present
if [ -f "$ROOT/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$ROOT/venv/bin/activate"
fi

echo "Starting backend on http://localhost:8000 ..."
cd "$ROOT"
python -m uvicorn api.main:app --reload --port 8000 &
BACKEND_PID=$!

echo "Starting frontend on http://localhost:5173 ..."
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

# Open browser after a short delay (best-effort — silently ignored if unavailable)
(sleep 4 && xdg-open http://localhost:5173 2>/dev/null || open http://localhost:5173 2>/dev/null || true) &

trap 'echo "Shutting down..."; kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true' EXIT INT TERM

wait "$BACKEND_PID" "$FRONTEND_PID"
