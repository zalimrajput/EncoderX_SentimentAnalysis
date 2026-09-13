/**
 * Typed client for the EncoderX FastAPI backend.
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface PredictionResult {
  sentiment: string;
  confidence: number;
  probabilities: Record<string, number>;
  inference_time_ms: number;
  mode: "model" | "demo";
}

export interface ModelInfo {
  model_name: string;
  labels: string[];
  max_length: number;
  device: string;
  mode: string;
  num_parameters: number | null;
}

/**
 * Pull a human-readable message out of a FastAPI error response.
 * FastAPI validation errors return `detail` as an array of issue objects.
 */
async function extractError(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail.length > 0) {
      const first = detail[0] as { msg?: string; loc?: (string | number)[] };
      const field = first.loc?.slice(1).join(".");
      return first.msg
        ? `${field ? `${field}: ` : ""}${first.msg}`
        : res.statusText;
    }
  } catch {
    /* fall through to status text */
  }
  return res.statusText || `Request failed with status ${res.status}`;
}

function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  return fetch(`${API_BASE_URL}${path}`, { signal }).then(async (res) => {
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`API error ${res.status}: ${body || res.statusText}`);
    }
    return res.json() as Promise<T>;
  });
}

export async function predictSentiment(
  review: string,
  signal?: AbortSignal,
): Promise<PredictionResult> {
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ review }),
    signal,
  });

  if (!res.ok) {
    throw new Error(await extractError(res));
  }

  return res.json() as Promise<PredictionResult>;
}

export function getModelInfo(signal?: AbortSignal): Promise<ModelInfo> {
  return get<ModelInfo>("/model-info", signal);
}

export function checkHealth(signal?: AbortSignal): Promise<{ status: string }> {
  return get<{ status: string }>("/health", signal);
}
