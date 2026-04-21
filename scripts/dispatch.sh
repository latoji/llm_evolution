#!/usr/bin/env bash
# dispatch.sh — watches PLAN.md and spawns fresh claude agents when tracks become AVAILABLE.
#
# Replaces sleeping/polling agents: each agent conversation lives only as long as its track.
# No idle context accumulation, no wasted tokens.
#
# Usage (from project root, venv activated):
#   bash scripts/dispatch.sh            # background processes + log files (default)
#   bash scripts/dispatch.sh --tmux     # one tmux window per agent (recommended)
#   bash scripts/dispatch.sh --wt       # one Windows Terminal tab per agent
#
# Requirements:
#   - claude CLI on PATH  (claude --version to verify)
#   - For --tmux: tmux installed in WSL  (sudo apt install tmux)
#   - For --wt:   Windows Terminal installed; run from WSL with access to wt.exe

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAN="$PROJECT_ROOT/PLAN.md"
LOG_DIR="$PROJECT_ROOT/logs"
POLL_INTERVAL=30
MODE="${1:-}"         # --tmux | --wt | (empty = background)
TMUX_SESSION="llm-build"

declare -A dispatched=()

log() { echo "[dispatch $(date '+%H:%M:%S')] $*"; }

# ── helpers ──────────────────────────────────────────────────────────────────

get_model() {
  # Returns the model ID for a given track heading, e.g. "Track 0"
  grep -A2 "^#### ${1}" "$PLAN" \
    | grep -oP '(?<=\*\*Model:\*\* `)claude-[a-z0-9-]+(?=`)'
}

available_tracks() {
  # Print heading of every track whose Status line contains AVAILABLE
  awk '
    /^#### Track / {
      heading = $0
      gsub(/^#### /, "", heading)
      gsub(/ —.*/, "", heading)
    }
    /Status.*AVAILABLE/ { print heading }
  ' "$PLAN"
}

all_complete() {
  local not_done
  not_done=$(grep -cE 'Status.*\[ (BLOCKED|IN PROGRESS|AVAILABLE)' "$PLAN" || true)
  [[ "$not_done" -eq 0 ]] && echo "done" || echo "running"
}

make_prompt() {
  local track="$1" model="$2" ts
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  cat <<PROMPT
Read \\\\wsl.localhost\\Ubuntu\\home\\latoj\\warp_projects\\LLM_prototype\\PLAN.md in full.

You are a ${model} agent. Follow the Agent Coordination Protocol in that file exactly.

Only claim tracks where the Model field says ${model}. Do not claim tracks designated for other models.

Claim ${track} now: it is AVAILABLE and its dependencies are COMPLETE.
Edit its Status line in PLAN.md to:
  [ IN PROGRESS — Agent: ${model}, Started: ${ts} ]
Save immediately. Wait 3 seconds. Re-read PLAN.md to confirm your claim.
Then read its spec file from \\\\wsl.localhost\\Ubuntu\\home\\latoj\\warp_projects\\LLM_prototype\\specs\\ and build it.

When all tests pass, mark Status as:
  [ COMPLETE — ${ts} ]
and exit.
PROMPT
}

# ── launch strategies ─────────────────────────────────────────────────────────

launch_background() {
  local track="$1" model="$2" prompt="$3"
  local safe="${track// /_}"
  local logfile="$LOG_DIR/agent_${safe}.log"
  mkdir -p "$LOG_DIR"
  claude --model "$model" --print "$prompt" > "$logfile" 2>&1 &
  log "  PID $! — tail -f logs/agent_${safe}.log"
}

launch_tmux() {
  local track="$1" model="$2" prompt="$3"
  local safe="${track// /_}"
  local logfile="$LOG_DIR/agent_${safe}.log"
  mkdir -p "$LOG_DIR"

  # Create session on first use
  if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux new-session -d -s "$TMUX_SESSION" -n "dispatcher"
    log "Created tmux session '$TMUX_SESSION'"
  fi

  local cmd="cd '$PROJECT_ROOT' && source venv/bin/activate 2>/dev/null || true && claude --model '$model' --print '$(printf '%s' "$prompt" | sed "s/'/'\\\\''/g")' 2>&1 | tee '$logfile'; echo '[agent exited]'; read"
  tmux new-window -t "$TMUX_SESSION" -n "$safe" "$cmd"
  log "  tmux window '$safe' opened — attach with: tmux attach -t $TMUX_SESSION"
}

launch_wt() {
  local track="$1" model="$2" prompt="$3"
  local safe="${track// /_}"
  local logfile="$LOG_DIR/agent_${safe}.log"
  mkdir -p "$LOG_DIR"

  # wt.exe is in PATH from WSL when Windows Terminal is installed
  local wsl_cmd="cd '$PROJECT_ROOT' && source venv/bin/activate 2>/dev/null || true && claude --model '$model' --print '$(printf '%s' "$prompt" | sed "s/'/'\\\\''/g")' 2>&1 | tee '$logfile'"
  wt.exe new-tab --title "$safe" wsl.exe bash -c "$wsl_cmd" &
  log "  Windows Terminal tab '$safe' opened"
}

# ── main loop ─────────────────────────────────────────────────────────────────

case "$MODE" in
  --tmux)     LAUNCH_FN="launch_tmux" ;;
  --wt)       LAUNCH_FN="launch_wt" ;;
  ""|--bg)    LAUNCH_FN="launch_background" ;;
  *)          echo "Unknown mode: $MODE. Use --tmux, --wt, or leave empty."; exit 1 ;;
esac

log "Dispatcher started (mode=${LAUNCH_FN#launch_}). Polling every ${POLL_INTERVAL}s."
log "Watching: $PLAN"

while true; do
  if [[ "$(all_complete)" == "done" ]]; then
    log "All tracks COMPLETE. Build finished. Dispatcher exiting."
    exit 0
  fi

  while IFS= read -r track; do
    [[ -z "$track" ]] && continue
    [[ -n "${dispatched[$track]+x}" ]] && continue

    model=$(get_model "$track") || { log "No model found for '$track', skipping."; continue; }
    dispatched["$track"]=1

    prompt=$(make_prompt "$track" "$model")
    log "Dispatching: track='$track' model='$model'"
    "$LAUNCH_FN" "$track" "$model" "$prompt"

  done < <(available_tracks)

  sleep "$POLL_INTERVAL"
done
