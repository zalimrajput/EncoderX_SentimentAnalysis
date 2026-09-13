"use client";

import { PredictionResult } from "@/lib/api";

interface ResultCardProps {
  result: PredictionResult;
  onClear: () => void;
}

const MAX_BAR_MS = 600; // cap the latency bar so it always reads well

export default function ResultCard({ result, onClear }: ResultCardProps) {
  const isPositive = result.sentiment.toLowerCase().startsWith("pos");
  const isDemo = result.mode === "demo";

  const probEntries = Object.entries(result.probabilities).sort(([, a], [, b]) => b - a);
  const latencyPct = Math.min(100, (result.inference_time_ms / MAX_BAR_MS) * 100);

  return (
    <section
      aria-live="polite"
      className={`animate-[fadeUp_0.5s_ease-out] overflow-hidden rounded-2xl border shadow-2xl backdrop-blur ${
        isPositive
          ? "border-emerald-500/30 bg-emerald-950/30 shadow-emerald-950/40"
          : "border-rose-500/30 bg-rose-950/30 shadow-rose-950/40"
      }`}
    >
      <div
        className={`h-1.5 w-full ${
          isPositive
            ? "bg-gradient-to-r from-emerald-500 via-teal-400 to-emerald-500"
            : "bg-gradient-to-r from-rose-500 via-red-400 to-rose-500"
        }`}
      />

      <div className="space-y-6 p-6 sm:p-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
              Analysis Result
            </p>
            <div className="mt-2 flex items-center gap-3">
              <span className="text-4xl" aria-hidden>
                {isPositive ? "🎉" : "👎"}
              </span>
              <div>
                <h2
                  className={`text-3xl font-bold tracking-tight sm:text-4xl ${
                    isPositive ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {result.sentiment}
                </h2>
                <p className="text-sm text-slate-400">
                  Predicted sentiment of your movie review
                </p>
              </div>
            </div>
          </div>

          <div className="text-right">
            <p
              className={`text-4xl font-bold tabular-nums sm:text-5xl ${
                isPositive ? "text-emerald-300" : "text-rose-300"
              }`}
            >
              {(result.confidence * 100).toFixed(1)}
              <span className="text-2xl">%</span>
            </p>
            <p className="text-xs uppercase tracking-wider text-slate-400">
              Confidence
            </p>
          </div>
        </div>

        {/* Confidence bar */}
        <div>
          <div className="mb-1.5 flex items-center justify-between text-xs text-slate-400">
            <span>Model confidence</span>
            <span className="tabular-nums">{(result.confidence * 100).toFixed(2)}%</span>
          </div>
          <div
            className="h-3 w-full overflow-hidden rounded-full bg-slate-800"
            role="progressbar"
            aria-valuenow={Math.round(result.confidence * 100)}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Model confidence"
          >
            <div
              className={`h-full rounded-full ${
                isPositive
                  ? "bg-gradient-to-r from-emerald-500 to-teal-300"
                  : "bg-gradient-to-r from-rose-500 to-red-300"
              }`}
              style={{
                width: `${result.confidence * 100}%`,
                transition: "width 0.9s cubic-bezier(0.22, 1, 0.36, 1)",
              }}
            />
          </div>
        </div>

        {/* Per-class probabilities */}
        <div className="grid gap-3 sm:grid-cols-2">
          {probEntries.map(([label, prob]) => (
            <div
              key={label}
              className="rounded-xl border border-slate-700/60 bg-slate-900/60 p-4"
            >
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium text-slate-300">{label}</span>
                <span className="font-semibold tabular-nums text-slate-100">
                  {(prob * 100).toFixed(2)}%
                </span>
              </div>
              <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                <div
                  className={`h-full rounded-full ${
                    label.toLowerCase().startsWith("pos")
                      ? "bg-emerald-400"
                      : "bg-rose-400"
                  }`}
                  style={{
                    width: `${prob * 100}%`,
                    transition: "width 0.9s cubic-bezier(0.22, 1, 0.36, 1)",
                  }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Metadata footer */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-800 pt-4 text-xs text-slate-400">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
            <span>
              ⚡ Inference:{" "}
              <span className="font-medium text-slate-200">
                {result.inference_time_ms.toFixed(1)} ms
              </span>
            </span>
            {isDemo && (
              <span
                className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 font-medium text-amber-300"
                title="Fine-tuned weights not found on the server — heuristic demo is active."
              >
                ⚠ Demo mode
              </span>
            )}
          </div>
          <button
            onClick={onClear}
            className="rounded-lg border border-slate-700 px-3 py-1.5 font-medium text-slate-300 transition-colors hover:border-slate-500 hover:bg-slate-800 hover:text-white"
          >
            Clear result
          </button>
        </div>
      </div>
    </section>
  );
}
