/** ModelCard — one panel per language model on the Stats page.
 *
 * Shows: model name, architecture note, accuracy graph, last generated sample.
 */
import { AccuracyGraph } from "./AccuracyGraph";
import type { AccuracyPoint } from "../types";
import { getModelMeta } from "../constants";

export interface ModelCardProps {
  name: string;
  points: AccuracyPoint[];
  lastText?: string;
  lastAccuracy?: number;
}

export function ModelCard({ name, points, lastText, lastAccuracy }: ModelCardProps) {
  const meta = getModelMeta(name);
  const latestAcc =
    lastAccuracy ??
    (points.length > 0 ? (points[points.length - 1]?.accuracy ?? null) : null);

  return (
    <div
      className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm"
      data-testid={`model-card-${name}`}
    >
      <div className="flex items-start justify-between gap-2 mb-1">
        <h3 className="font-semibold text-slate-800">{meta.displayName}</h3>
        {latestAcc !== null && (
          <span className="text-sm font-mono bg-blue-50 text-blue-700 px-2 py-0.5 rounded">
            {(latestAcc * 100).toFixed(1)}%
          </span>
        )}
      </div>

      <p className="text-xs text-slate-500 italic mb-3">{meta.architectureNote}</p>

      <AccuracyGraph points={points} />

      {lastText && (
        <div className="mt-3 border-t border-slate-100 pt-3">
          <p className="text-xs text-slate-400 mb-1">Last generated sample</p>
          <p className="font-mono text-xs text-slate-600 leading-relaxed break-words line-clamp-3">
            {lastText}
          </p>
        </div>
      )}
    </div>
  );
}
