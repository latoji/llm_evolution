/** useProgressSocket — subscribes to the shared WS connection.
 *
 * The `onEvent` callback is wrapped in a ref so callers do not need
 * to memoize it — the subscription is created once and always calls
 * the latest version of the callback.
 */
import { useEffect, useRef } from "react";
import { useWSContext } from "../context/WSContext";
import type { WSMessage } from "../types";

export interface UseProgressSocketResult {
  connected: boolean;
}

export function useProgressSocket(
  onEvent: (e: WSMessage) => void
): UseProgressSocketResult {
  const { connected, subscribe } = useWSContext();
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  useEffect(() => {
    const unsub = subscribe((e) => onEventRef.current(e));
    return unsub;
  }, [subscribe]);

  return { connected };
}
