/** Fetch wrappers for all backend HTTP endpoints.
 *
 * All URLs are derived from VITE_API_BASE (defaults to localhost:8000).
 * Every function throws on non-OK responses.
 */
import type {
  IngestUploadResponse,
  IngestStatusResponse,
  IngestPauseResponse,
  GenerateResponse,
  AccuracyHistoryResponse,
  LastOutputResponse,
  DBResetResponse,
  TablePageResponse,
  RowCounts,
} from "./types";

export const API_BASE: string =
  import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export const WS_BASE: string = API_BASE.replace(/^https?/, "ws");

async function checkOk(r: Response): Promise<void> {
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    throw new Error(text || `HTTP ${r.status.toString()}`);
  }
}

export async function uploadFiles(
  files: File[]
): Promise<IngestUploadResponse> {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  const r = await fetch(`${API_BASE}/ingest/upload`, {
    method: "POST",
    body: fd,
  });
  await checkOk(r);
  return r.json();
}

export async function pauseIngest(): Promise<IngestPauseResponse> {
  const r = await fetch(`${API_BASE}/ingest/pause`, { method: "POST" });
  await checkOk(r);
  return r.json();
}

export async function getStatus(): Promise<IngestStatusResponse> {
  const r = await fetch(`${API_BASE}/ingest/status`);
  await checkOk(r);
  return r.json();
}

export async function getAccuracy(): Promise<AccuracyHistoryResponse> {
  const r = await fetch(`${API_BASE}/stats/accuracy`);
  await checkOk(r);
  return r.json();
}

export async function getLastGens(): Promise<LastOutputResponse> {
  const r = await fetch(`${API_BASE}/stats/last_generations`);
  await checkOk(r);
  return r.json();
}

export async function generate(
  word_count: number,
  auto_correct: boolean
): Promise<GenerateResponse> {
  const r = await fetch(`${API_BASE}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ word_count, auto_correct }),
  });
  await checkOk(r);
  return r.json();
}

export async function getTables(): Promise<{ tables: string[] }> {
  const r = await fetch(`${API_BASE}/db/tables`);
  await checkOk(r);
  return r.json();
}

export async function getTable(
  name: string,
  limit = 100,
  offset = 0
): Promise<TablePageResponse> {
  const r = await fetch(
    `${API_BASE}/db/table/${name}?limit=${limit}&offset=${offset}`
  );
  await checkOk(r);
  return r.json();
}

export async function getRowCounts(): Promise<RowCounts> {
  const r = await fetch(`${API_BASE}/db/row_counts`);
  await checkOk(r);
  return r.json();
}

export async function resetDB(): Promise<DBResetResponse> {
  const r = await fetch(`${API_BASE}/db/reset`, { method: "POST" });
  await checkOk(r);
  return r.json();
}
