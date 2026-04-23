/** IngestContext — holds all ingest progress state at app level.
 *
 * Subscribes to the shared WebSocket once (via WSContext) so state is never
 * lost when the user navigates away from the Ingest page.
 */
import {
  createContext,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useWSContext } from "./WSContext";
import { useQueryClient } from "@tanstack/react-query";
import { ACCURACY_QUERY_KEY } from "../hooks/useAccuracy";
import { getStatus } from "../api";
import type { WSMessage } from "../types";

export interface CompletedRun {
  model: string;
  text: string;
  accuracy: number;
}

export interface IngestState {
  totalChunks: number;
  completedChunks: number;
  operation: string | null;
  operationPct: number;
  isRunning: boolean;
  isPaused: boolean;
  activeMCModel: string | null;
  liveTokens: string;
  completedRuns: CompletedRun[];
  eventLog: WSMessage[];
  clearLog: () => void;
}

export const IngestContext = createContext<IngestState | null>(null);

export function IngestProvider({ children }: { children: ReactNode }) {
  const { subscribe } = useWSContext();
  const queryClient = useQueryClient();

  const [totalChunks, setTotalChunks] = useState(0);
  const [completedChunks, setCompletedChunks] = useState(0);
  const [operation, setOperation] = useState<string | null>(null);
  const [operationPct, setOperationPct] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [activeMCModel, setActiveMCModel] = useState<string | null>(null);
  const [liveTokens, setLiveTokens] = useState("");
  const [completedRuns, setCompletedRuns] = useState<CompletedRun[]>([]);
  const [eventLog, setEventLog] = useState<WSMessage[]>([]);

  // Bootstrap from the API on mount so state is correct even if training
  // was already running before this component mounted.
  useEffect(() => {
    getStatus()
      .then((s) => {
      if (s.state === "running" || s.state === "paused") {
          setIsRunning(s.state === "running");
          setIsPaused(s.state === "paused");
          if (s.total_chunks != null) setTotalChunks(s.total_chunks);
          if (s.current_chunk != null) setCompletedChunks(s.current_chunk);
        }
      })
      .catch(() => {/* non-fatal — WS events will update state */});
  }, []);

  // Buffer events to avoid a re-render per WebSocket frame
  const liveTokensRef = useRef("");
  const eventBufferRef = useRef<WSMessage[]>([]);

  useEffect(() => {
    const id = setInterval(() => {
      if (eventBufferRef.current.length > 0) {
        const pending = eventBufferRef.current.splice(0);
        setEventLog((prev) => [...prev, ...pending].slice(-200));
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
          setIsPaused(false);
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
          setIsPaused(false);
          setOperation(null);
          setOperationPct(0);
          break;
      }
    },
    [queryClient]
  );

  useEffect(() => subscribe(handleEvent), [subscribe, handleEvent]);

  const clearLog = useCallback(() => {
    setEventLog([]);
    eventBufferRef.current = [];
  }, []);

  return (
    <IngestContext.Provider
      value={{
        totalChunks,
        completedChunks,
        operation,
        operationPct,
        isRunning,
        isPaused,
        activeMCModel,
        liveTokens,
        completedRuns,
        eventLog,
        clearLog,
      }}
    >
      {children}
    </IngestContext.Provider>
  );
}

