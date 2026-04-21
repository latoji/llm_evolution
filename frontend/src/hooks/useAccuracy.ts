/** useAccuracy — fetches and caches model accuracy history.
 *
 * Polls every 2 seconds while the tab is in focus.
 */
import { useQuery } from "@tanstack/react-query";
import { getAccuracy } from "../api";
import type { AccuracyHistoryResponse } from "../types";

export const ACCURACY_QUERY_KEY = ["accuracy"] as const;

export function useAccuracy() {
  return useQuery<AccuracyHistoryResponse, Error>({
    queryKey: ACCURACY_QUERY_KEY,
    queryFn: getAccuracy,
    refetchInterval: 2000,
    refetchIntervalInBackground: false,
  });
}
