#!/usr/bin/env bash
# watch_tier.sh — per-model-tier watcher for Option B (Warp Launch Configurations).
#
# Each Warp pane runs one instance of this script for its model tier.
# It polls PLAN.md, and when a track for its tier becomes AVAILABLE it runs
# a fresh claude agent inline (output visible directly in the pane), then
# loops back to polling. Exits when all tracks for this tier are COMPLETE.
#
# Usage (run inside WSL, venv activated):
#   bash scripts/watch_tier.sh claude-sonnet-4-6
#   bash scripts/watch_tier.sh claude-opus-4-6
#   bash scripts/watch_tier.sh claude-opus-4-7

set -euo pipefail

MODEL="${1:?Usage: watch_tier.sh <model-id>}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLAN="$PROJECT_ROOT/PLAN.md"
POLL_INTERVAL=30

declare -A done_tracks=()

log()  { echo; echo "──────────────────────────────────────────────"; echo "  [${MODEL}] $(date '+%H:%M:%S')  $*"; echo "──────────────────────────────────────────────"; echo; }
info() { echo "  [${MODEL}] $*"; }

# ── helpers ───────────────────────────────────────────────────────────────────

# Tracks assigned to this tier whose Status is AVAILABLE
available_for_tier() {
  awk -v model="$MODEL" '
    /^#### Track / {
      heading = $0
      gsub(/^#### /, "", heading)
      gsub(/ —.*/, "", heading)
    }
    /\*\*Model:\*\*/ && $0 ~ model { has_model = 1 }
    /\*\*Model:\*\*/ && $0 !~ model { has_model = 0 }
    has_model && /Status.*AVAILABLE/ { print heading }
  ' "$PLAN"
}

# All tracks for this tier — returns count not yet COMPLETE
tier_remaining() {
  awk -v model="$MODEL" '
    /^#### Track / { in_block = 0 }
    /\*\*Model:\*\*/ && $0 ~ model { in_block = 1 }
    in_block && /Status/ && !/COMPLETE/ { count++ }
    END { print count+0 }
  ' "$PLAN"
}

make_prompt() {
  local track="$1" ts
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  cat <<PROMPT
Read \\\\wsl.localhost\\Ubuntu\\home\\latoj\\warp_projects\\LLM_prototype\\PLAN.md in full.

You are a ${MODEL} agent. Follow the Agent Coordination Protocol in that file exactly.

Only claim tracks where the Model field says ${MODEL}. Do not claim tracks designated for other models.

Claim ${track} now: it is AVAILABLE and its dependencies are COMPLETE.
Edit its Status line in PLAN.md to:
  [ IN PROGRESS — Agent: ${MODEL}, Started: ${ts} ]
Save immediately. Wait 3 seconds. Re-read PLAN.md to confirm your claim.
Then read its spec file from \\\\wsl.localhost\\Ubuntu\\home\\latoj\\warp_projects\\LLM_prototype\\specs\\ and build it.
Follow the Coding Rules in PLAN.md Section 5 (max 400 lines/file, no bare Any, Lucide icons on frontend).

When all tests pass, mark Status as:
  [ COMPLETE — $(date -u +%Y-%m-%dT%H:%M:%SZ) ]
and exit.
PROMPT
}

# ── main loop ─────────────────────────────────────────────────────────────────

clear
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  LLM Evolution — Tier Watcher                           ║"
echo "║  Model : ${MODEL}"
echo "║  Watching : PLAN.md every ${POLL_INTERVAL}s                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo

while true; do
  remaining=$(tier_remaining)
  if [[ "$remaining" -eq 0 ]]; then
    log "All ${MODEL} tracks COMPLETE. Pane done — safe to close."
    exit 0
  fi

  while IFS= read -r track; do
    [[ -z "$track" ]] && continue
    [[ -n "${done_tracks[$track]+x}" ]] && continue

    done_tracks["$track"]=1
    log "Track unlocked: ${track}"
    info "Spawning fresh claude agent…"
    echo

    prompt=$(make_prompt "$track")
    # Run claude inline — output streams directly into this pane
    claude --model "$MODEL" --print "$prompt"

    echo
    info "Agent exited for ${track}. Returning to poll loop."
  done < <(available_for_tier)

  info "Polling… (next check in ${POLL_INTERVAL}s)"
  sleep "$POLL_INTERVAL"
done
