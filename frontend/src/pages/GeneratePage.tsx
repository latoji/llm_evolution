/** GeneratePage — generate text from all 13 models simultaneously. */
import { useState } from "react";
import { RefreshCw } from "lucide-react";
import { generate } from "../api";
import { RealWordText } from "../components/RealWordText";
import { getModelMeta, sortedModelNames } from "../constants";
import type { GenerateResponse, ModelOutput } from "../types";

export function GeneratePage() {
  const [wordCount, setWordCount] = useState(50);
  const [autoCorrect, setAutoCorrect] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GenerateResponse | null>(null);

  async function handleGenerate() {
    setLoading(true);
    setError(null);
    try {
      const data = await generate(wordCount, autoCorrect);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setLoading(false);
    }
  }

  // Sort outputs by model display order
  const sortedOutputs: ModelOutput[] = result
    ? sortedModelNames(
        result.outputs.map((o) => o.model_name),
        false
      )
        .map(
          (name) =>
            result.outputs.find((o) => o.model_name === name) as ModelOutput
        )
        .filter(Boolean)
    : [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-slate-800">Generate Text</h1>

      {/* Controls */}
      <div className="bg-white border border-slate-200 rounded-lg p-4 flex flex-wrap items-end gap-4">
        <div>
          <label className="block text-sm text-slate-600 mb-1">
            Word count
            <input
              type="number"
              min={10}
              max={500}
              value={wordCount}
              onChange={(e) => setWordCount(Number(e.target.value))}
              className="ml-2 w-24 border border-slate-300 rounded px-2 py-1 text-sm"
            />
          </label>
        </div>

        <label className="flex items-center gap-2 text-sm text-slate-600 cursor-pointer">
          <input
            type="checkbox"
            checked={autoCorrect}
            onChange={(e) => setAutoCorrect(e.target.checked)}
            className="rounded"
          />
          Augment (spell / grammar correct)
        </label>

        <button
          data-testid="generate-btn"
          onClick={() => void handleGenerate()}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          {loading ? "Generating…" : "Generate"}
        </button>
      </div>

      {error && (
        <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded p-3">
          {error}
        </p>
      )}

      {/* Results — 13 model panels */}
      {result && (
        <div className="space-y-4">
          {sortedOutputs.map((output) => {
            const meta = getModelMeta(output.model_name);
            const words = output.word_results.map(([w, real]) => ({ w, real }));
            return (
              <div
                key={output.model_name}
                className="bg-white border border-slate-200 rounded-lg p-4"
                data-testid={`model-card-${output.model_name}`}
              >
                <div className="flex items-start justify-between gap-2 mb-1">
                  <h3 className="font-semibold text-slate-800">{meta.displayName}</h3>
                  <span className="text-sm font-mono bg-blue-50 text-blue-700 px-2 py-0.5 rounded shrink-0">
                    Real words: {(output.real_word_pct * 100).toFixed(1)}%
                  </span>
                </div>
                <p className="text-xs text-slate-500 italic mb-3">
                  {meta.architectureNote}
                </p>
                <RealWordText words={words} />
                {output.corrected_text && (
                  <div className="mt-2 pt-2 border-t border-slate-100">
                    <p className="text-xs text-slate-400 mb-1">Corrected</p>
                    <p className="font-mono text-xs text-slate-600">{output.corrected_text}</p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {!result && !loading && (
        <div className="text-center py-16 text-slate-400 text-sm">
          Click <strong>Generate</strong> to run all 13 models simultaneously.
        </div>
      )}
    </div>
  );
}
