/** NavBar — top navigation bar for the four app pages.
 *
 * Displays a WebSocket connectivity dot and a persistent ingest progress
 * indicator that remains visible on every page while training is running.
 */
import { NavLink } from "react-router-dom";
import { useWSContext } from "../context/WSContext";
import { useIngestContext } from "../context/useIngestContext";

const NAV_LINKS = [
  { to: "/ingest", label: "Ingest" },
  { to: "/stats", label: "Stats" },
  { to: "/generate", label: "Generate" },
  { to: "/db", label: "DB" },
  { to: "/help", label: "Help" },
] as const;

function linkClass({ isActive }: { isActive: boolean }): string {
  return isActive
    ? "font-semibold text-blue-600 underline underline-offset-4"
    : "text-slate-600 hover:text-blue-500";
}

export function NavBar() {
  const { connected } = useWSContext();
  const { isRunning, isPaused, completedChunks, totalChunks, operation } = useIngestContext();
  const isActive = isRunning || isPaused;
  const chunkPct = totalChunks > 0 ? Math.round((completedChunks / totalChunks) * 100) : 0;

  return (
    <nav className="bg-white border-b border-slate-200 px-6 py-3 flex items-center gap-6">
      <span className="font-bold text-slate-800 mr-4 text-lg">LLM Evolution</span>
      {NAV_LINKS.map(({ to, label }) => (
        <NavLink key={to} to={to} className={linkClass}>
          {label}
        </NavLink>
      ))}

      <div className="ml-auto flex items-center gap-4">
        {/* Ingest progress — visible on every page while training or paused */}
        {isActive && totalChunks > 0 && (
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <div className="w-32 h-1.5 bg-slate-100 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  isPaused ? "bg-amber-400" : "bg-blue-500"
                }`}
                style={{ width: `${chunkPct}%` }}
              />
            </div>
            <span className="tabular-nums whitespace-nowrap">
              {completedChunks}/{totalChunks}
              {operation ? ` · ${operation.split(' ')[0]}…` : ""}
            </span>
          </div>
        )}

        {/* WebSocket status dot */}
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <span
            className={`inline-block w-2.5 h-2.5 rounded-full ${
              connected ? "bg-green-500" : "bg-red-400"
            }`}
            title={connected ? "WebSocket connected" : "WebSocket disconnected"}
          />
          <span>{connected ? "Live" : "Reconnecting…"}</span>
        </div>
      </div>
    </nav>
  );
}
