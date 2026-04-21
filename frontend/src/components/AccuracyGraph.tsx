/** AccuracyGraph — line chart of accuracy over chunk_id for one model.
 *
 * Uses Recharts. x-axis = chunk index, y-axis = accuracy (0–1).
 */
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { AccuracyPoint } from "../types";

export interface AccuracyGraphProps {
  points: AccuracyPoint[];
}

interface ChartDatum {
  chunk: number;
  acc: number;
}

export function AccuracyGraph({ points }: AccuracyGraphProps) {
  if (points.length === 0) {
    return (
      <div className="flex items-center justify-center h-20 text-sm text-slate-400">
        No data yet
      </div>
    );
  }

  const data: ChartDatum[] = points.map((p) => ({
    chunk: p.chunk_id,
    acc: Math.round(p.accuracy * 1000) / 1000,
  }));

  return (
    <ResponsiveContainer width="100%" height={120}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis
          dataKey="chunk"
          tick={{ fontSize: 10 }}
          label={{ value: "chunk", position: "insideBottomRight", offset: -4, fontSize: 9 }}
        />
        <YAxis
          domain={[0, 1]}
          tick={{ fontSize: 10 }}
          width={32}
          tickFormatter={(v: number) => v.toFixed(1)}
        />
        <Tooltip
          formatter={(v) => {
            const val = typeof v === "number" ? v : 0;
            return [`${(val * 100).toFixed(1)}%`, "accuracy"] as [string, string];
          }}
          labelFormatter={(l) => `chunk ${String(l)}`}
        />
        <Line
          type="monotone"
          dataKey="acc"
          stroke="#3b82f6"
          dot={false}
          strokeWidth={2}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
