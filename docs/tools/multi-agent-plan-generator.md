# MULTI-AGENT PLAN GENERATOR: PLAN.md

This document serves as the central orchestration hub for multiple AI agents working in parallel. It leverages git-based state locking to prevent race conditions and ensures a seamless integration phase.

## 1. AGENT COORDINATION PROTOCOL
This plan is the single source of truth. Every agent **must** follow this protocol exactly.

### Rules for Every Agent
1.  **Always start with a fresh pull:** Run `git pull` before reading this file. Never work from a stale copy.
2.  **Claim your task atomically:**
    - Find the first task whose Status is `[ AVAILABLE ]` and whose dependencies are all `[ COMPLETE ]`.
    - Edit this file: Set Status to `[ IN PROGRESS ]`, fill in your Agent name and Started timestamp.
    - **Commit and push immediately** before writing a single line of code.
    - Message format: `chore: claim Track [X] — [agent: ID]`
    - *Note:* If your push is rejected, someone beat you to it. Pull, re-read the board, and pick a different task.
3.  **Work strictly within your boundaries:** Only edit files listed in your track's Deliverables. Do not touch files owned by other tracks.
4.  **Mark complete:** When all deliverables are done and verified (no TS/lint errors):
    - Edit this file: Set Status to `[ COMPLETE ]` and fill in the Completed timestamp.
    - Commit and push all work in one final commit.
    - Message format: `feat: complete Track [X] — [agent: ID]

Co-Authored-By: Warp`
5.  **The Polling Loop:** After completing your track, immediately `git pull`. Scan the board for the next available task. If nothing is available, poll every 2 minutes (`git pull` and re-scan).
6.  **Integration Trigger:** Only stop working when Track Z is `[ COMPLETE ]` or claimed by another agent.
7.  **Surface Blockers:** If a dependency is delayed, set status to `[ BLOCKED — waiting on: Track X ]` and commit immediately.

---

## 2. TASK BOARD

### Status Legend
- `[ AVAILABLE ]` — Ready to claim. Dependencies are met.
- `[ IN PROGRESS — Agent: , Started:  ]` — Claimed and being worked on.
- `[ COMPLETE —  ]` — Done and pushed.
- `[ BLOCKED — waiting on: Track X ]` — Dependency not yet met.

### Tracks

#### Track 0 — Shared Contracts
- **Status:** `[ AVAILABLE ]`
- **Depends on:** None
- **Agent:** - **Started:** - **Completed:** - **Deliverables:**
    - `src/types/database.ts`: Shared TypeScript interfaces and schema definitions.
    - `src/constants/index.ts`: Shared constants, API endpoints, and configuration.
    - `tailwind.config.js`: Design tokens and theme configuration.

#### Track A — Backend / API Services
- **Status:** `[ BLOCKED — waiting on: Track 0 ]`
- **Depends on:** Track 0
- **Agent:** - **Started:** - **Completed:** - **Deliverables:**
    - `src/lib/api-client.ts`: Core API fetch wrapper.
    - `src/services/data-service.ts`: Backend logic and data fetching methods.

#### Track B — Authentication & User State
- **Status:** `[ BLOCKED — waiting on: Track 0 ]`
- **Depends on:** Track 0
- **Agent:** - **Started:** - **Completed:** - **Deliverables:**
    - `src/hooks/useAuth.ts`: Hook for managing user session and login state.
    - `src/components/auth/LoginForm.tsx`: Authentication UI.

#### Track C — Core UI Components
- **Status:** `[ BLOCKED — waiting on: Track 0 ]`
- **Depends on:** Track 0
- **Agent:** - **Started:** - **Completed:** - **Deliverables:**
    - `src/components/ui/`: Atomic components (Buttons, Inputs, Modals).
    - `src/components/layout/`: Global Shell, Sidebar, and Navigation components.

#### Track Z — Integration
- **Status:** `[ BLOCKED — waiting on: ALL TRACKS ]`
- **Depends on:** Track 0, A, B, C
- **Agent:** - **Started:** - **Completed:** - **Deliverables:**
    - `src/App.tsx`: Main entry point wiring all components and providers.
    - `README.md`: Final documentation update.
    - Full project type-check and lint verification.

---

## 3. TECHNICAL SPECIFICATION

### System Architecture
- **Framework:** Next.js / Vite (Strict TypeScript)
- **Styling:** Tailwind CSS
- **State:** React Context or Zustand
- **Database/Auth:** Supabase / Firebase

### Data Flow
1. **Track 0** establishes the types all other agents must import.
2. **Track A** provides the methods for data persistence.
3. **Track B** handles the identity layer.
4. **Track C** builds the visual building blocks.
5. **Track Z** mounts the application and performs final QA.

### Coding Rules
- **Modularity:** Maximum 400 lines per file.
- **Strict Typing:** No `any` types allowed.
- **Icons:** Use Lucide-react.
