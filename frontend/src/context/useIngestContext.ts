/** useIngestContext — accessor hook for IngestContext.
 *
 * Kept in its own file so Vite Fast Refresh can hot-swap it
 * without invalidating IngestContext.tsx (which exports a component).
 */
import { useContext } from "react";
import { IngestContext, type IngestState } from "./IngestContext";

export type { IngestState };

export function useIngestContext(): IngestState {
  const ctx = useContext(IngestContext);
  if (!ctx) throw new Error("useIngestContext must be used inside IngestProvider");
  return ctx;
}
