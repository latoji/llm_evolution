/** useIngestStatus — polls the ingest status endpoint.
 *
 * Automatically slows down polling when ingest is idle.
 */
import { useQuery } from "@tanstack/react-query";
import { getStatus } from "../api";
import type { IngestStatusResponse } from "../types";

export const INGEST_STATUS_QUERY_KEY = ["ingestStatus"] as const;

export function useIngestStatus() {
  return useQuery<IngestStatusResponse, Error>({
    queryKey: INGEST_STATUS_QUERY_KEY,
    queryFn: getStatus,
    refetchInterval: (query) => {
      const state = query.state.data?.state;
      // Poll faster while ingest is actively running or paused
      return state === "running" || state === "paused" ? 1000 : 5000;
    },
    refetchIntervalInBackground: false,
  });
}
