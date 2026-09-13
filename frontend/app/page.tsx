"use client";

import { useCallback, useRef, useState } from "react";
import { predictSentiment, PredictionResult } from "@/lib/api";
import ResultCard from "@/components/ResultCard";

const MAX_CHARS = 10_000;

const EXAMPLE_REVIEWS: { label: string; emoji: string; text: string }[] = [
  {
    label: "Positive review",
    emoji: "🌟",
    text: "This movie was absolutely fantastic! The storyline was gripping from start to finish, the performances were outstanding, and the cinematography was breathtaking. I was completely immersed the entire time and left the theater feeling inspired. Highly recommend it to anyone who loves great filmmaking.",
  },
  {
    label: "Negative review",
    emoji: "💀",
    text: "A complete waste of two hours. The plot was predictable and full of clichés, the dialogue was painfully awkward, and the pacing dragged terribly in the middle. Even the usually reliable lead actor seemed bored. I kept waiting for it to get better, but it never did. Definitely not worth watching.",
  },
  {
    label: "Mixed/subtle",
    emoji: "🎭",
    text: "The film has its moments — a couple of genuinely beautiful scenes and a strong opening act — but the second half falls apart with a rushed, unsatisfying ending. Beautiful to look at, yet hollow where it matters most.",
  },
];

export default function Home() {
  const [review, setReview] = useState("");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const charCount = review.length;
  const nearLimit = charCount > MAX_CHARS * 0.9;

  const analyze = useCallback(async () => {
    const text = review.trim();
    if (!text || isLoading) return;

    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      const prediction = await predictSentiment(text);
      setResult(prediction);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while contacting the sentiment API.",
      );
    } finally {
      setIsLoading(false);
    }
  }, [review, isLoading]);

  const clearAll = useCallback(() => {
    setReview("");
    setResult(null);
    setError(null);
    textareaRef.current?.focus();
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      void analyze();
    }
  };

  return (
    <main className="relative mx-auto w-full max-w-3xl flex-1 px-4 py-10 sm:px-6 lg:px-8">
      {/* Hero */}
      <header className="mb-10 text-center">
        <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-slate-700/70 bg-slate-900/70 px-4 py-1.5 text-xs font-medium text-slate-300 backdrop-blur">
          <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
          Fine-tuned BERT · IMDB Movie Reviews
        </div>
        <h1 className="bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-4xl font-extrabold tracking-tight text-transparent sm:text-5xl">
          EncoderX Sentiment Analysis
        </h1>
        <p className="mx-auto mt-3 max-w-2xl text-base leading-relaxed text-slate-400 sm:text-lg">
          Enter a movie review below and the fine-tuned BERT model will classify it
          as <span className="font-medium text-emerald-400">Positive</span> or{" "}
          <span className="font-medium text-rose-400">Negative</span> — with a
          confidence score.
        </p>
      </header>

      <div className="space-y-6">
        {/* Input card */}
        <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 shadow-xl backdrop-blur sm:p-8">
          <label
            htmlFor="review"
            className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400"
          >
            Movie Review
          </label>

          <div className="relative mt-3">
            <textarea
              id="review"
              ref={textareaRef}
              value={review}
              onChange={(e) => setReview(e.target.value.slice(0, MAX_CHARS))}
              onKeyDown={handleKeyDown}
              rows={7}
              placeholder="e.g. The film was a masterclass in tension — every scene pulled me deeper into the story. Highly recommended!"
              className="w-full resize-y rounded-xl border border-slate-700 bg-slate-950/70 p-4 text-base leading-relaxed text-slate-100 placeholder:text-slate-500 focus:border-emerald-500/60 focus:outline-none focus:ring-2 focus:ring-emerald-500/20"
            />
            <span
              className={`pointer-events-none absolute bottom-3 right-3 rounded-md bg-slate-900/90 px-2 py-0.5 text-xs tabular-nums ${
                nearLimit ? "text-amber-400" : "text-slate-500"
              }`}
            >
              {charCount.toLocaleString()} / {MAX_CHARS.toLocaleString()}
            </span>
          </div>

          {/* Example reviews */}
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-500">
              Try an example:
            </span>
            {EXAMPLE_REVIEWS.map((ex) => (
              <button
                key={ex.label}
                type="button"
                onClick={() => {
                  setReview(ex.text);
                  setResult(null);
                  setError(null);
                  textareaRef.current?.focus();
                }}
                className="rounded-full border border-slate-700 bg-slate-800/60 px-3 py-1 text-xs font-medium text-slate-300 transition-colors hover:border-emerald-500/50 hover:bg-emerald-500/10 hover:text-emerald-300"
              >
                {ex.emoji} {ex.label}
              </button>
            ))}
          </div>

          {/* Actions */}
          <div className="mt-6 flex flex-col gap-3 sm:flex-row">
            <button
              onClick={() => void analyze()}
              disabled={isLoading || !review.trim()}
              className="group relative flex h-12 flex-1 items-center justify-center gap-2 overflow-hidden rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 font-semibold text-white shadow-lg shadow-emerald-950/50 transition-all hover:shadow-emerald-900/50 hover:brightness-110 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:brightness-100"
            >
              {isLoading ? (
                <>
                  <span
                    className="h-5 w-5 animate-spin rounded-full border-2 border-white/30 border-t-white"
                    aria-hidden
                  />
                  Analyzing…
                </>
              ) : (
                <>✨ Analyze Sentiment</>
              )}
            </button>
            <button
              onClick={clearAll}
              disabled={isLoading || (!review && !result && !error)}
              className="h-12 rounded-xl border border-slate-700 px-6 font-medium text-slate-300 transition-colors hover:border-slate-500 hover:bg-slate-800 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
            >
              Clear
            </button>
          </div>

          {error && (
            <p
              role="alert"
              className="mt-4 flex items-start gap-2 rounded-xl border border-rose-500/40 bg-rose-500/10 p-3 text-sm text-rose-300"
            >
              <span aria-hidden>⚠️</span>
              <span>{error}</span>
            </p>
          )}
        </section>

        {/* Result */}
        {result && <ResultCard result={result} onClear={clearAll} />}

        {isLoading && (
          <div className="animate-[fadeUp_0.3s_ease-out] space-y-3 rounded-2xl border border-slate-800 bg-slate-900/60 p-6 backdrop-blur">
            <div className="flex items-center gap-3 text-sm text-slate-400">
              <span
                className="h-4 w-4 animate-spin rounded-full border-2 border-emerald-500/30 border-t-emerald-400"
                aria-hidden
              />
              Running BERT inference on your review…
            </div>
            <div className="space-y-2 pt-1">
              <div className="h-3 w-2/3 animate-pulse rounded bg-slate-800" />
              <div className="h-3 w-1/2 animate-pulse rounded bg-slate-800" />
              <div className="h-24 w-full animate-pulse rounded-xl bg-slate-800/70" />
            </div>
          </div>
        )}
      </div>

      <footer className="mt-12 border-t border-slate-800 pt-6 text-center text-xs text-slate-500">
         BERT Fine-Tuning &amp; Sentiment Analysis 
      </footer>
    </main>
  );
}
