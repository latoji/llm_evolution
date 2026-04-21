# Multi-Agent Launch Prompts

Three prompts — one per model tier. Paste the matching prompt into each fresh Claude Desktop window at the start of a build session. Every agent reads the same `PLAN.md`; the tier discipline is enforced by the instructions in the prompt itself.

All three prompts reference the same absolute path:

```
\\wsl.localhost\Ubuntu\home\latoj\warp_projects\LLM_prototype\PLAN.md
```

---

## Prompt 1 — `claude-sonnet-4-6` agent

> Read `\\wsl.localhost\Ubuntu\home\latoj\warp_projects\LLM_prototype\PLAN.md` in full.
>
> You are a `claude-sonnet-4-6` agent. Follow the Agent Coordination Protocol in that file exactly.
>
> Only claim tracks where the `Model` field says `claude-sonnet-4-6`. Do not claim tracks designated for `claude-opus-4-6` or `claude-opus-4-7`, even if they are available and your own tracks are complete. When all `claude-sonnet-4-6`-designated tracks are COMPLETE or IN PROGRESS, stop polling and exit.
>
> **Claim protocol:** When you claim a track, edit its `Status` line in `PLAN.md` to:
> `[ IN PROGRESS — Agent: claude-sonnet-4-6, Started: <ISO timestamp> ]`
> Save immediately. Wait 3 seconds. Re-read `PLAN.md`. If your claim is still there, proceed. If overwritten, pick a different available track in your tier.
>
> **Completion protocol:** When tests pass, set `Status` to:
> `[ COMPLETE — <ISO timestamp> ]`
> and fill in your `Completed` field with the same timestamp.
>
> Begin now: check the Task Board, find the first AVAILABLE track designated for `claude-sonnet-4-6` whose dependencies are all COMPLETE, claim it atomically, read its spec file at `\\wsl.localhost\Ubuntu\home\latoj\warp_projects\LLM_prototype\specs\`, and build it.

---

## Prompt 2 — `claude-opus-4-6` agent

> Read `\\wsl.localhost\Ubuntu\home\latoj\warp_projects\LLM_prototype\PLAN.md` in full.
>
> You are a `claude-opus-4-6` agent. Follow the Agent Coordination Protocol in that file exactly.
>
> Only claim tracks where the `Model` field says `claude-opus-4-6`. Do not claim tracks designated for `claude-opus-4-7` or `claude-sonnet-4-6`, even if they are available and your own tracks are complete. When all `claude-opus-4-6`-designated tracks are COMPLETE or IN PROGRESS, stop polling and exit.
>
> **Claim protocol:** When you claim a track, edit its `Status` line in `PLAN.md` to:
> `[ IN PROGRESS — Agent: claude-opus-4-6, Started: <ISO timestamp> ]`
> Save immediately. Wait 3 seconds. Re-read `PLAN.md`. If your claim is still there, proceed. If overwritten, pick a different available track in your tier.
>
> **Completion protocol:** When tests pass, set `Status` to:
> `[ COMPLETE — <ISO timestamp> ]`
> and fill in your `Completed` field with the same timestamp.
>
> Begin now: check the Task Board, find the first AVAILABLE track designated for `claude-opus-4-6` whose dependencies are all COMPLETE, claim it atomically, read its spec file at `\\wsl.localhost\Ubuntu\home\latoj\warp_projects\LLM_prototype\specs\`, and build it.

---

## Prompt 3 — `claude-opus-4-7` agent

> Read `\\wsl.localhost\Ubuntu\home\latoj\warp_projects\LLM_prototype\PLAN.md` in full.
>
> You are a `claude-opus-4-7` agent. Follow the Agent Coordination Protocol in that file exactly.
>
> Only claim tracks where the `Model` field says `claude-opus-4-7`. Do not claim tracks designated for `claude-opus-4-6` or `claude-sonnet-4-6`, even if they are available and your own tracks are complete. When all `claude-opus-4-7`-designated tracks are COMPLETE or IN PROGRESS, stop polling and exit.
>
> **Claim protocol:** When you claim a track, edit its `Status` line in `PLAN.md` to:
> `[ IN PROGRESS — Agent: claude-opus-4-7, Started: <ISO timestamp> ]`
> Save immediately. Wait 3 seconds. Re-read `PLAN.md`. If your claim is still there, proceed. If overwritten, pick a different available track in your tier.
>
> **Completion protocol:** When tests pass, set `Status` to:
> `[ COMPLETE — <ISO timestamp> ]`
> and fill in your `Completed` field with the same timestamp.
>
> Begin now: check the Task Board, find the first AVAILABLE track designated for `claude-opus-4-7` whose dependencies are all COMPLETE, claim it atomically, read its spec file at `\\wsl.localhost\Ubuntu\home\latoj\warp_projects\LLM_prototype\specs\`, and build it.

---

## Launch method — Option B (Warp Launch Configurations)

These prompts are embedded in `scripts/watch_tier.sh` and fired automatically.
**You do not paste these prompts manually.** The workflow is:

### One-time setup
1. Copy `scripts/warp_launch.yaml` to `%APPDATA%\Warp\launch_configurations\llm_build.yaml` on the Windows side
2. Verify claude CLI: open a WSL terminal in Warp and run `claude --version`

### Starting a build session
1. In Warp: click the Warp icon (top-left) → **Launch Configurations** → **llm_build**
2. Four panes open automatically:
   - **Top-left**: Sonnet 4.6 watcher (Tracks 0, A, B1, E)
   - **Top-right**: Opus 4.6 watcher (Tracks C, D2, Z)
   - **Bottom-left**: Opus 4.7 watcher (Tracks B2, D1)
   - **Bottom-right**: Live build status (auto-refreshes every 15s)
3. The Sonnet pane immediately finds Track 0 AVAILABLE and starts. Others poll until their dependencies clear.

### What happens automatically
- Sonnet → Track 0 → (on complete) → A → B1 → E
- Opus 4.7 → waits → B2 (after Track 0) → D1 (after C)
- Opus 4.6 → waits → C (after A+B1+B2) → D2 (after D1) → Z (after D2+E)
- Each pane prints the live agent output, then shows "Polling…" between tracks
- Each pane exits automatically when its tier's tracks are all COMPLETE

### Track-to-tier mapping

| Tier | Tracks |
| :--- | :--- |
| `claude-sonnet-4-6` | 0, A, B1, E |
| `claude-opus-4-6` | C, D2, Z |
| `claude-opus-4-7` | B2, D1 |

### Locking note
Locking is file-based (no git remote): edit Status → save → wait 3s → re-read to confirm claim. If a local git repo is later initialized, add `git add PLAN.md && git commit -m "chore: claim Track X — agent: <id>"` after each Status change for stronger guarantees.
