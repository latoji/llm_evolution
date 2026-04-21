/** DBPage — browse DuckDB tables with pagination and a reset button. */
import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";
import { getTables, getTable, getRowCounts, resetDB } from "../api";
import type { TableRow } from "../types";

const TABLES_KEY = ["db-tables"] as const;
const ROW_COUNTS_KEY = ["db-row-counts"] as const;

function tableKey(name: string, offset: number) {
  return ["db-table", name, offset] as const;
}

export function DBPage() {
  const queryClient = useQueryClient();
  const [activeTable, setActiveTable] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [resetMsg, setResetMsg] = useState<string | null>(null);
  const [resetError, setResetError] = useState<string | null>(null);

  const LIMIT = 100;

  const { data: tablesData, isLoading: tablesLoading } = useQuery({
    queryKey: TABLES_KEY,
    queryFn: getTables,
  });

  const { data: rowCounts } = useQuery({
    queryKey: ROW_COUNTS_KEY,
    queryFn: getRowCounts,
  });

  const tables = tablesData?.tables ?? [];
  const selectedTable = activeTable ?? tables[0] ?? null;

  const { data: tableData, isLoading: tableLoading } = useQuery({
    queryKey: tableKey(selectedTable ?? "", offset),
    queryFn: () => getTable(selectedTable!, LIMIT, offset),
    enabled: selectedTable !== null,
  });

  function handleTableSelect(name: string) {
    setActiveTable(name);
    setOffset(0);
  }

  async function handleReset() {
    if (!window.confirm("Reset the entire database? This cannot be undone.")) return;
    setResetMsg(null);
    setResetError(null);
    try {
      const res = await resetDB();
      setResetMsg(res.message);
      await queryClient.invalidateQueries();
    } catch (err) {
      setResetError(err instanceof Error ? err.message : "Reset failed");
    }
  }

  const totalRows = selectedTable ? (rowCounts?.[selectedTable] ?? tableData?.total ?? 0) : 0;
  const columns: string[] =
    tableData && tableData.rows.length > 0
      ? Object.keys(tableData.rows[0] as TableRow)
      : [];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h1 className="text-2xl font-bold text-slate-800">Database</h1>
        <button
          data-testid="db-reset-btn"
          onClick={() => void handleReset()}
          className="flex items-center gap-1 px-3 py-1.5 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200 font-medium"
        >
          <Trash2 className="w-4 h-4" />
          Reset DB
        </button>
      </div>

      {resetMsg && (
        <p className="text-sm text-green-600 bg-green-50 border border-green-200 rounded p-2">
          {resetMsg}
        </p>
      )}
      {resetError && (
        <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">
          {resetError}
        </p>
      )}

      {tablesLoading && (
        <p className="text-sm text-slate-400 italic">Loading tables…</p>
      )}

      {/* Table tabs */}
      {tables.length > 0 && (
        <div className="flex flex-wrap gap-1 border-b border-slate-200 pb-2">
          {tables.map((t) => (
            <button
              key={t}
              onClick={() => handleTableSelect(t)}
              className={`px-3 py-1.5 text-sm rounded-t font-medium transition-colors ${
                selectedTable === t
                  ? "bg-blue-600 text-white"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              }`}
            >
              {t}
              {rowCounts?.[t] !== undefined && (
                <span className="ml-1 text-xs opacity-70">({rowCounts[t]})</span>
              )}
            </button>
          ))}
        </div>
      )}

      {/* Table content */}
      {selectedTable && (
        <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
          <div className="px-4 py-2 border-b border-slate-100 text-sm text-slate-600 flex items-center justify-between">
            <span>
              <strong>{selectedTable}</strong> — {totalRows} rows
            </span>
            <span className="text-xs text-slate-400">
              showing {offset + 1}–{Math.min(offset + LIMIT, totalRows)}
            </span>
          </div>

          {tableLoading && (
            <p className="text-sm text-slate-400 italic p-4">Loading…</p>
          )}

          {!tableLoading && columns.length > 0 && (
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs text-slate-700">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr>
                    {columns.map((col) => (
                      <th
                        key={col}
                        className="px-3 py-2 text-left font-semibold text-slate-600 whitespace-nowrap"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {tableData!.rows.map((row, i) => (
                    <tr key={i} className="hover:bg-slate-50">
                      {columns.map((col) => (
                        <td
                          key={col}
                          className="px-3 py-1.5 font-mono whitespace-nowrap max-w-xs truncate"
                          title={String(row[col] ?? "")}
                        >
                          {String(row[col] ?? "")}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {!tableLoading && (tableData?.rows.length ?? 0) === 0 && (
            <p className="text-sm text-slate-400 italic p-4">No rows.</p>
          )}

          {/* Pagination */}
          {totalRows > LIMIT && (
            <div className="flex items-center justify-between px-4 py-2 border-t border-slate-100 text-sm">
              <button
                onClick={() => setOffset((v) => Math.max(0, v - LIMIT))}
                disabled={offset === 0}
                className="px-3 py-1 bg-slate-100 rounded hover:bg-slate-200 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              <button
                onClick={() => setOffset((v) => v + LIMIT)}
                disabled={offset + LIMIT >= totalRows}
                className="px-3 py-1 bg-slate-100 rounded hover:bg-slate-200 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Load more
              </button>
            </div>
          )}
        </div>
      )}

      {!tablesLoading && tables.length === 0 && (
        <div className="text-center py-16 text-slate-400 text-sm">
          No tables found. Ingest a corpus to populate the database.
        </div>
      )}
    </div>
  );
}
