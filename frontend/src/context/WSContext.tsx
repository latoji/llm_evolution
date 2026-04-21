/** WebSocket context — opens a single WS connection and fans events out to all subscribers.
 *
 * Place <WSProvider> at the app root so every page shares one connection.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { wsMessageSchema, type WSMessage } from "../types";
import { WS_BASE } from "../api";

const WS_URL = `${WS_BASE}/ws/progress`;
const RECONNECT_DELAY_MS = 3000;

type EventCallback = (e: WSMessage) => void;

interface WSContextValue {
  connected: boolean;
  subscribe: (cb: EventCallback) => () => void;
}

const WSContext = createContext<WSContextValue | null>(null);

export function WSProvider({ children }: { children: ReactNode }) {
  const [connected, setConnected] = useState(false);
  const callbacksRef = useRef<Set<EventCallback>>(new Set());
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (
      wsRef.current?.readyState === WebSocket.OPEN ||
      wsRef.current?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onclose = () => {
      setConnected(false);
      reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
    };

    ws.onerror = () => {
      ws.close();
    };

    ws.onmessage = (e: MessageEvent<string>) => {
      let raw: unknown;
      try {
        raw = JSON.parse(e.data) as unknown;
      } catch {
        return; // ignore non-JSON frames
      }
      const result = wsMessageSchema.safeParse(raw);
      if (!result.success) return; // ignore unknown/ping frames
      const msg = result.data;
      callbacksRef.current.forEach((cb) => cb(msg));
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current !== null) {
        clearTimeout(reconnectTimerRef.current);
      }
      wsRef.current?.close();
    };
  }, [connect]);

  const subscribe = useCallback((cb: EventCallback) => {
    callbacksRef.current.add(cb);
    return () => {
      callbacksRef.current.delete(cb);
    };
  }, []);

  return (
    <WSContext.Provider value={{ connected, subscribe }}>
      {children}
    </WSContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useWSContext(): WSContextValue {
  const ctx = useContext(WSContext);
  if (!ctx) throw new Error("useWSContext must be used inside WSProvider");
  return ctx;
}
