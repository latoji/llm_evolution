/** StatsPage — shows all 13 model accuracy histories and last generated samples. */
import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useAccuracy, ACCURACY_QUERY_KEY } from "../hooks/useAccuracy";
import { useProgressSocket } from "../hooks/useProgressSocket";
import { getLastGens } from "../api";
import { ModelCard } from "../components/ModelCard";
import { MODEL_META, sortedModelNames } from "../constants";
import type { WSMessage } from "../types";

const LAST_GENS_KEY = ["lastGens"] as const;

export function StatsPage() {
  const queryClient = useQueryClient();
  const [sortByAccuracy, setSortByAccuracy] = useState(false);

  const { data: accuracyData, isLoading } = useAccuracy();

  const { data: lastGens } = useQuery({
    queryKey: LAST_GENS_KEY,
    queryFn: getLastGens,
    refetchInterval: 5000,
    refetchIntervalInBackground: false,
  });

  // Invalidate both caches when any MC evaluation completes
  useProgressSocket((e: WSMessage) => {
    if (e.type === "mc_complete") {
      void queryClient.invalidateQueries({ queryKey: ACCURACY_QUERY_KEY });
      void queryClient.invalidateQueries({ queryKey: LAST_GENS_KEY });
    }
  });

  const models = accuracyData?.models ?? {};

  // Build per-model latest accuracy for sort
  const latestAccuracy: Record<string, number> = Object.fromEntries(
    Object.entries(models).map(([name, points]) => [
      name,
      points[points.length - 1]?.accuracy ?? 0,
    ])
  );

  // Always show all 13 known models; add any server-returned extras
  const allNames = Array.from(
    new Set([...Object.keys(MODEL_META), ...Object.keys(models)])
  );
  const displayNames = sortedModelNames(
    allNames,
    sortByAccuracy,
    latestAccuracy
  );

  const lastGensMap = Object.fromEntries(
    (lastGens?.outputs ?? []).map((o) => [o.model_name, o])
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h1 className="text-2xl font-bold text-slate-800">Model Stats</h1>
        <label className="flex items-center gap-2 text-sm text-slate-600 cursor-pointer">
          <input
            type="checkbox"
            checked={sortByAccuracy}
            onChange={(e) => setSortByAccuracy(e.target.checked)}
            className="rounded"
          />
          Sort by accuracy
        </label>
      </div>

      {isLoading && (
        <p className="text-sm text-slate-400 italic">Loading accuracy data…</p>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {displayNames.map((name) => (
          <ModelCard
            key={name}
            name={name}
            points={models[name] ?? []}
            lastText={lastGensMap[name]?.text}
            lastAccuracy={lastGensMap[name]?.real_word_pct}
          />
        ))}
      </div>
    </div>
  );
}
