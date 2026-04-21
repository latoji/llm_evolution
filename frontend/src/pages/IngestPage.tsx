/** IngestPage — upload corpus files and watch all 13 models train in real time. */
import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronUp, Pause } from "lucide-react";
import { uploadFiles, pauseIngest } from "../api";
import { useProgressSocket } from "../hooks/useProgressSocket";
import { ACCURACY_QUERY_KEY } from "../hooks/useAccuracy";
import type { WSMessage } from "../types";

interface CompletedRun {
  model: string;
  text: string;
  accuracy: number;
}

export function IngestPage() {
  const queryClient = useQueryClient();

  // Ingest progress state
  const [totalChunks, setTotalChunks] = useState(0);
  const [completedChunks, setCompletedChunks] = useState(0);
  const [operation, setOperation] = useState<string | null>(null);
  const [operationPct, setOperationPct] = useState(0);
  const [isRunning, setIsRunning] = useState(false);

  // MC generation panel
  const liveTokensRef = useRef("");
  const [liveTokens, setLiveTokens] = useState("");
  const [activeMCModel, setActiveMCModel] = useState<string | null>(null);
  const [completedRuns, setCompletedRuns] = useState<CompletedRun[]>([]);

  // Event log — buffered to avoid thrashing at high event rates
  const eventBufferRef = useRef<WSMessage[]>([]);
  const [eventLog, setEventLog] = useState<WSMessage[]>([]);
  const [showLog, setShowLog] = useState(false);

  // Upload / drop zone
  const [isDragging, setIsDragging] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);

  // Flush event buffer every 200 ms to keep the log from triggering per-event renders
  useEffect(() => {
    const id = setInterval(() => {
      if (eventBufferRef.current.length > 0) {
        const pending = eventBufferRef.current.splice(0);
        setEventLog((prev) => [...prev, ...pending].slice(-100));
      }
    }, 200);
    return () => clearInterval(id);
  }, []);

  const handleEvent = useCallback(
    (e: WSMessage) => {
      eventBufferRef.current.push(e);

      switch (e.type) {
        case "chunk_start":
          setTotalChunks(e.total_chunks);
          setOperation(e.operation);
          setOperationPct(0);
          setIsRunning(true);
          // Reset per-ingest counters on first chunk
          if (e.chunk_index === 0) setCompletedChunks(0);
          break;
        case "chunk_progress":
          setOperation(e.operation);
          setOperationPct(e.pct);
          break;
        case "chunk_done":
          setCompletedChunks((prev) => prev + 1);
          break;
        case "mc_token":
          setActiveMCModel(e.model);
          liveTokensRef.current += e.token;
          setLiveTokens(liveTokensRef.current);
          break;
        case "mc_complete":
          setCompletedRuns((prev) => [
            { model: e.model, text: liveTokensRef.current, accuracy: e.accuracy },
            ...prev.slice(0, 9),
          ]);
          liveTokensRef.current = "";
          setLiveTokens("");
          void queryClient.invalidateQueries({ queryKey: ACCURACY_QUERY_KEY });
          break;
        case "ingest_complete":
          setIsRunning(false);
          setOperation(null);
          setOperationPct(0);
          break;
      }
    },
    [queryClient]
  );

  useProgressSocket(handleEvent);

  async function doUpload(files: File[]) {
    setUploadError(null);
    setUploadMsg(null);
    try {
      const res = await uploadFiles(files);
      setUploadMsg(
        `Accepted ${res.accepted_files.length} file(s), ${res.total_chunks} chunks queued.`
      );
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Upload failed");
    }
  }

  function onDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragging(false);
    const dropped = Array.from(e.dataTransfer.files).filter((f) =>
      f.name.endsWith(".txt")
    );
    if (dropped.length) void doUpload(dropped);
  }

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (files.length) void doUpload(files);
  }

  async function handlePause() {
    try {
      await pauseIngest();
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Pause failed");
    }
  }

  const chunkPct =
    totalChunks > 0 ? Math.round((completedChunks / totalChunks) * 100) : 0;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-slate-800">Ingest Corpus</h1>

      {/* Drop zone */}
      <div
        data-testid="ingest-dropzone"
        onDrop={onDrop}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        className={`border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-colors ${
          isDragging
            ? "border-blue-500 bg-blue-50"
            : "border-slate-300 hover:border-blue-400 bg-white"
        }`}
      >
        <p className="text-slate-500 mb-3">Drop <code>.txt</code> files here</p>
        <label className="cursor-pointer inline-block bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 text-sm">
          Browse files
          <input
            type="file"
            multiple
            accept=".txt"
            className="hidden"
            onChange={onFileChange}
          />
        </label>
        {uploadMsg && <p className="mt-3 text-green-600 text-sm">{uploadMsg}</p>}
        {uploadError && <p className="mt-3 text-red-600 text-sm">{uploadError}</p>}
      </div>

      {/* Operation banner + pause */}
      {operation && (
        <div className="bg-white border border-slate-200 rounded-lg p-4 flex items-center gap-4">
          <div className="flex-1">
            <p className="text-sm font-medium text-slate-700">
              {operation}&hellip; {operationPct}%
            </p>
            <div className="mt-1 h-2 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full transition-all"
                style={{ width: `${operationPct}%` }}
              />
            </div>
          </div>
          <button
            data-testid="pause-btn"
            onClick={() => void handlePause()}
            disabled={!isRunning}
            className="flex items-center gap-1 px-3 py-1.5 text-sm bg-amber-100 text-amber-700 rounded hover:bg-amber-200 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Pause className="w-4 h-4" />
            Pause
          </button>
        </div>
      )}

      {/* Chunk progress bar */}
      {totalChunks > 0 && (
        <div className="bg-white border border-slate-200 rounded-lg p-4">
          <p className="text-sm text-slate-600 mb-1">
            Chunks: {completedChunks} / {totalChunks} ({chunkPct}%)
          </p>
          <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500 rounded-full transition-all"
              style={{ width: `${chunkPct}%` }}
            />
          </div>
        </div>
      )}

      {/* Live MC generation panel */}
      {(activeMCModel ?? completedRuns.length > 0) && (
        <div className="bg-white border border-slate-200 rounded-lg p-4 space-y-3">
          <h2 className="font-semibold text-slate-700 text-sm uppercase tracking-wide">
            Monte Carlo Generation
          </h2>
          {activeMCModel && (
            <div>
              <p className="text-xs text-slate-500 mb-1">
                Generating — <strong>{activeMCModel}</strong>
              </p>
              <p className="font-mono text-xs text-slate-700 bg-slate-50 rounded p-2 min-h-6 break-words">
                {liveTokens}
                <span className="animate-pulse">▌</span>
              </p>
            </div>
          )}
          {completedRuns.slice(0, 5).map((run, i) => (
            <div key={i} className="border-t border-slate-100 pt-2">
              <p className="text-xs text-slate-500">
                <strong>{run.model}</strong> — {(run.accuracy * 100).toFixed(1)}% real words
              </p>
              <p className="font-mono text-xs text-slate-600 break-words line-clamp-2">
                {run.text}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Event log */}
      <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
        <button
          className="w-full flex items-center justify-between px-4 py-3 text-sm text-slate-600 hover:bg-slate-50"
          onClick={() => setShowLog((v) => !v)}
        >
          <span>Event log ({eventLog.length} events)</span>
          {showLog ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
        {showLog && (
          <div className="max-h-64 overflow-y-auto border-t border-slate-100 font-mono text-xs text-slate-600 p-2 space-y-0.5">
            {eventLog.length === 0 && (
              <p className="text-slate-400 italic p-2">No events yet.</p>
            )}
            {eventLog.map((ev, i) => (
              <div key={i} className="truncate">
                <span className="text-blue-500 mr-2">{ev.type}</span>
                {JSON.stringify(ev)}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
