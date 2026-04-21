# Track E — Frontend (React + Vite)

**Model**: `claude-sonnet-4-6`
**Dependencies**: Track D2 complete (backend routes stable)

## Scope

Build the four pages of the educational UI: **Ingest**, **Stats**, **Generate**, **DB**. Vite + React + TypeScript. Tailwind for styling. Recharts for the Stats graph. Single-page app with top navigation.

The frontend is deliberately simple. No state-management library (Redux/Zustand) — React Query for server state, local `useState` for component state.

## Upstream dependencies

- All Track D2 endpoints (`/ingest/*`, `/stats/*`, `/generate`, `/db/*`, `/ws/progress`)
- `api/contracts.py` — the TypeScript types in `src/types.ts` are a hand-port of this file

## Downstream consumers

- **Track Z** (integration) runs a Playwright smoke test against the built app

## Files owned

```
frontend/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── types.ts                    (ports api/contracts.py)
│   ├── api.ts                      (fetch wrappers + WS hook)
│   ├── hooks/
│   │   ├── useProgressSocket.ts
│   │   ├── useAccuracy.ts
│   │   └── useIngestStatus.ts
│   ├── components/
│   │   ├── NavBar.tsx
│   │   ├── RealWordText.tsx        (reusable green/red word renderer)
│   │   ├── AccuracyGraph.tsx
│   │   └── ModelCard.tsx
│   └── pages/
│       ├── IngestPage.tsx
│       ├── StatsPage.tsx
│       ├── GeneratePage.tsx
│       └── DBPage.tsx
└── tests/
    └── e2e.spec.ts                 (Playwright, run from Track Z)
```

## Files you must NOT modify

- Anything outside `frontend/`
- `api/contracts.py` — if types diverge, update `src/types.ts`, not the backend

---

## Stack choices (non-negotiable)

| Concern | Choice | Why |
|---|---|---|
| Bundler | Vite 5 | Fast HMR, minimal config |
| Language | TypeScript, strict mode | Catches shape drift between frontend and backend |
| Styling | Tailwind CSS | No invented class conventions; fast for a 4-page app |
| Server state | `@tanstack/react-query` | Caching + revalidation + loading states built-in |
| Routing | `react-router-dom` v6 | Trivial page structure |
| Charts | `recharts` | Declarative, React-native, good enough for line graphs |
| WebSocket | native `WebSocket` wrapped in a hook | No `socket.io` needed — backend speaks raw WS |

Do **not** add: Redux, MobX, Zustand, styled-components, CSS modules, MUI, Ant Design, Chakra. If you feel you need one of these, re-read the scope.

---

## Implementation

### `src/api.ts`

```typescript
const BASE = "http://localhost:8000";

export async function uploadFiles(files: File[]): Promise<{status: string; file_count: number}> {
  const fd = new FormData();
  files.forEach(f => fd.append("files", f));
  const r = await fetch(`${BASE}/ingest/upload`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export const pauseIngest    = () => fetch(`${BASE}/ingest/pause`, {method: "POST"}).then(r => r.json());
export const getStatus      = () => fetch(`${BASE}/ingest/status`).then(r => r.json());
export const getModels      = () => fetch(`${BASE}/stats/models`).then(r => r.json());
export const getAccuracy    = () => fetch(`${BASE}/stats/accuracy`).then(r => r.json());
export const getLastGens    = () => fetch(`${BASE}/stats/last_generations`).then(r => r.json());
export const generate       = (n_words: number, augment: boolean) =>
  fetch(`${BASE}/generate`, {method: "POST", headers: {"Content-Type": "application/json"},
                              body: JSON.stringify({n_words, augment})}).then(r => r.json());
export const getTables      = () => fetch(`${BASE}/db/tables`).then(r => r.json());
export const getTable       = (name: string, limit=100, offset=0) =>
  fetch(`${BASE}/db/table/${name}?limit=${limit}&offset=${offset}`).then(r => r.json());
export const getRowCounts   = () => fetch(`${BASE}/db/row_counts`).then(r => r.json());
export const resetDB        = () => fetch(`${BASE}/db/reset`, {method: "POST"}).then(r => r.json());
```

### `src/hooks/useProgressSocket.ts`

```typescript
import { useEffect, useRef, useState } from "react";
import type { WSMessage } from "../types";

export function useProgressSocket(onEvent: (e: WSMessage) => void) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/progress");
    wsRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (e) => {
      try { onEvent(JSON.parse(e.data)); } catch { /* ignore malformed */ }
    };
    return () => ws.close();
  }, [onEvent]);

  return { connected };
}
```

The hook is deliberately dumb — the page that consumes events holds the state machine. Events the caller must handle:

- `chunk_start`, `chunk_progress`, `chunk_done`
- `mc_model_start`, `mc_token`, `mc_complete`
- `file_rejected`, `ingest_paused`, `ingest_complete`
- `ping` (ignore)

### `src/pages/IngestPage.tsx` (~200 lines)

Layout (top to bottom):

1. **Drop zone** (or file picker) — accepts multiple `.txt` files. On drop, call `uploadFiles`.
2. **Current operation banner** — one line: "Training transformer… 62%", fed by the last `chunk_progress` event.
3. **Chunk progress bar** — chunks completed / total, filled based on `chunk_start`'s `total_chunks`.
4. **Live MC generation panel** — when `mc_model_start` fires, show the model name and an empty text box. Append each `mc_token` event's text. On `mc_complete`, color-code words green/red using `RealWordText` and freeze that run, then start the next.
5. **Pause button** — enabled only while ingest is running. Calls `pauseIngest`.
6. **Event log** — scrolling list of the last 100 events (collapsed by default for educational view).

The educational goal: a user watching this page should be able to narrate what the system is doing without reading code.

**Architecture note under each model card** (from SPEC.md §2): small info-panel with the model's mechanism in one sentence, e.g. *"Char 3-gram: predicts each character from the previous 2."*

### `src/pages/StatsPage.tsx` (~150 lines)

- Vertically scrollable list of 13 `ModelCard` components, one per model, sorted by `display_order` (default) or by accuracy desc (toggle via textbox/select).
- Each `ModelCard` contains:
  - Model name + architecture note
  - A `recharts` line chart: x = chunk_id, y = accuracy (0.0–1.0)
  - Last generated text rendered via `RealWordText` (green real, red fake)
- Auto-refresh every 2 seconds via React Query's `refetchInterval`, AND via `useProgressSocket` listening for `mc_complete` events that invalidate the cache.

### `src/pages/GeneratePage.tsx` (~120 lines)

- Number input for `n_words` (default 50, range 10–500)
- Checkbox for "Augment (spell/grammar correct)"
- "Generate" button — calls `generate(n_words, augment)`, disables while loading
- Results: 13 panels (same order as Stats), each showing:
  - Model name
  - Generated text via `RealWordText`
  - `Real words: XX%`

### `src/pages/DBPage.tsx` (~180 lines)

- Tab bar listing tables from `/db/tables`
- For the selected table:
  - Table of rows (first 100, with `Load more` for pagination via `offset`)
  - Row count shown above the table (from `/db/row_counts`)
- **Reset DB** button at the bottom — red, confirms via `window.confirm`, calls `resetDB`, then invalidates all React Query caches.

### `src/components/RealWordText.tsx`

```typescript
interface Props { words: {w: string; real: boolean}[] }

export function RealWordText({ words }: Props) {
  return (
    <p className="font-mono text-sm leading-relaxed">
      {words.map((x, i) => (
        <span key={i} className={x.real ? "text-green-600" : "text-red-600"}>
          {x.w}{" "}
        </span>
      ))}
    </p>
  );
}
```

Used by Stats and Generate pages.

### `src/App.tsx`

```typescript
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NavBar } from "./components/NavBar";
import { IngestPage, StatsPage, GeneratePage, DBPage } from "./pages";

const qc = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <div className="min-h-screen bg-slate-50">
          <NavBar />
          <main className="max-w-6xl mx-auto p-6">
            <Routes>
              <Route path="/" element={<Navigate to="/ingest" replace />} />
              <Route path="/ingest" element={<IngestPage />} />
              <Route path="/stats" element={<StatsPage />} />
              <Route path="/generate" element={<GeneratePage />} />
              <Route path="/db" element={<DBPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
```

### `src/components/NavBar.tsx`

Simple top bar with 4 `NavLink`s. Active tab gets a bold underline. No hamburger menu, no responsive drawer — desktop-only.

---

## Testing

Unit tests: skip. React component unit testing adds disproportionate overhead for a 4-page app.

End-to-end test (one file, owned by Track Z but invoked here): `tests/e2e.spec.ts` (Playwright). Track Z will write it; you only need to ensure your components have stable `data-testid` attributes on:

- File drop zone: `data-testid="ingest-dropzone"`
- Pause button: `data-testid="pause-btn"`
- Generate button: `data-testid="generate-btn"`
- DB reset button: `data-testid="db-reset-btn"`
- Each model card: `data-testid="model-card-{name}"` (e.g. `model-card-char_3gram`)

---

## Acceptance criteria

- [ ] `npm install && npm run dev` starts the Vite dev server on :5173
- [ ] `npm run build` produces a `dist/` bundle under 500 KB gzipped
- [ ] `npm run typecheck` (`tsc --noEmit`) passes with zero errors in strict mode
- [ ] All four pages render without console errors in a fresh Chromium
- [ ] WebSocket reconnects after backend restart (visible indicator in nav bar)
- [ ] Tailwind is configured; no inline hex colors in JSX except in chart config

---

## Pitfalls

- **Do not store WS events in React state one-at-a-time** for the event log. At 30+ events/sec you will thrash re-renders. Buffer into a ref and flush to state every ~200ms.
- **Recharts is SSR-unsafe** — fine for Vite client-only builds, but if you ever add SSR, wrap it in `dynamic`.
- **`fetch` does not throw on 4xx/5xx.** Check `r.ok` everywhere and surface backend error messages — the user needs to see "ingest already running" as a real message, not a silent no-op.
- **React Query's `refetchInterval` keeps running when the tab is backgrounded** unless you set `refetchIntervalInBackground: false`. Do set it — no need to burn CPU off-screen.
- **Do not put the WS connection in every page.** Open it once in `App.tsx` (or a context provider) and broadcast to pages via a simple event emitter or React Query's cache mutations. Multiple WS connections means multiple event streams — wasteful and hard to debug.
- **File drop zone must prevent the browser's default behavior** on `dragover` / `drop`, or the browser will navigate away from your app.
- **Tailwind purge/content config must include `./src/**/*.{ts,tsx}`** or classes will be stripped in production.
- **Never hardcode `http://localhost:8000`** across every file. Centralize in `src/api.ts` and read from `import.meta.env.VITE_API_BASE` with localhost as default.

---

## Model assignment

**Sonnet 4.6.** Standard React patterns, well-trodden stack. The interesting part is the WS event choreography on the Ingest page, which is pattern-work once the event types are known.
