/** TypeScript contracts — hand-port of api/contracts.py.
 *
 * HTTP response types are plain interfaces.
 * WebSocket message types use Zod schemas for runtime validation.
 */
import { z } from "zod";

// ---------------------------------------------------------------------------
// Ingest endpoints
// ---------------------------------------------------------------------------

export interface RejectedFile {
  filename: string;
  reason: string;
}

export interface IngestUploadResponse {
  accepted_files: string[];
  rejected_files: RejectedFile[];
  total_chunks: number;
}

export interface IngestStatusResponse {
  state: "idle" | "running" | "paused" | "complete";
  current_chunk: number | null;
  total_chunks: number | null;
  chunks_accepted: number;
  chunks_rejected: number;
}

export interface IngestPauseResponse {
  paused: boolean;
  message: string;
}

// ---------------------------------------------------------------------------
// Generation endpoints
// ---------------------------------------------------------------------------

export interface ModelOutput {
  model_name: string;
  raw_text: string;
  corrected_text: string | null;
  word_results: [string, boolean][];
  real_word_pct: number;
}

export interface GenerateResponse {
  outputs: ModelOutput[];
}

// ---------------------------------------------------------------------------
// Stats endpoints
// ---------------------------------------------------------------------------

export interface AccuracyPoint {
  chunk_id: number;
  accuracy: number;
  perplexity: number | null;
  timestamp: string;
}

export interface AccuracyHistoryResponse {
  models: Record<string, AccuracyPoint[]>;
}

export interface LastOutputEntry {
  model_name: string;
  text: string;
  real_word_pct: number;
}

export interface LastOutputResponse {
  outputs: LastOutputEntry[];
}

// ---------------------------------------------------------------------------
// DB viewer endpoints
// ---------------------------------------------------------------------------

export interface DBResetResponse {
  success: boolean;
  message: string;
}

export type TableRow = Record<string, unknown>;

export interface TablePageResponse {
  rows: TableRow[];
  total: number;
  page: number;
  page_size: number;
}

export type RowCounts = Record<string, number>;

// ---------------------------------------------------------------------------
// WebSocket messages — discriminated union, validated with Zod
// ---------------------------------------------------------------------------

const wsChunkStartSchema = z.object({
  type: z.literal("chunk_start"),
  chunk_index: z.number(),
  total_chunks: z.number(),
  operation: z.string(),
});

const wsChunkProgressSchema = z.object({
  type: z.literal("chunk_progress"),
  operation: z.string(),
  pct: z.number(),
});

const wsMCTokenSchema = z.object({
  type: z.literal("mc_token"),
  model: z.string(),
  token: z.string(),
  run: z.number(),
});

const wsMCCompleteSchema = z.object({
  type: z.literal("mc_complete"),
  model: z.string(),
  accuracy: z.number(),
  run: z.number(),
});

const wsChunkDoneSchema = z.object({
  type: z.literal("chunk_done"),
  chunk_index: z.number(),
  status: z.union([z.literal("accepted"), z.literal("rejected")]),
  accuracy_delta: z.record(z.string(), z.number()),
  reason: z.string().nullable(),
});

const wsIngestCompleteSchema = z.object({
  type: z.literal("ingest_complete"),
  chunks_accepted: z.number(),
  chunks_rejected: z.number(),
});

export const wsMessageSchema = z.discriminatedUnion("type", [
  wsChunkStartSchema,
  wsChunkProgressSchema,
  wsMCTokenSchema,
  wsMCCompleteSchema,
  wsChunkDoneSchema,
  wsIngestCompleteSchema,
]);

export type WSMessage = z.infer<typeof wsMessageSchema>;
