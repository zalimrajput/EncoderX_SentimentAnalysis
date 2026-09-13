"""EncoderX — BERT Sentiment Analysis API.

FastAPI service that loads a fine-tuned BERT model and exposes a POST /predict
endpoint for movie-review sentiment inference.

System flow:
    User Review -> Next.js Frontend -> FastAPI /predict -> Fine-tuned BERT
    -> Sentiment + Confidence -> Frontend Result

Run:
    uvicorn main:app --reload --port 8000

Configuration (environment variables):
    MODEL_DIR   Directory holding the fine-tuned model artifacts
                (config.json, tokenizer files, weights). Default: ./model
    MODEL_NAME  Display name used when the model is unavailable. Default: bert-base-uncased
    MAX_LENGTH  Maximum tokenized sequence length. Default: 256

If the fine-tuned weights cannot be found, the API still starts in a clearly
labelled "demo" mode using a lightweight keyword heuristic so the full-stack
flow can be demonstrated before the checkpoint is placed in MODEL_DIR.
"""

import os
import time

# This backend runs BERT with PyTorch. If TensorFlow happens to be installed in
# the environment, transformers would try to import it (and its broken protobuf
# deps could crash startup) — so explicitly disable the TF integration first.
os.environ.setdefault("USE_TF", "NO")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    TRANSFORMERS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
FALLBACK_MODEL_NAME = os.getenv("MODEL_NAME", "bert-base-uncased")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

API_TITLE = "EncoderX — BERT Sentiment Analysis API"
API_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class AppState:
    """Holds the loaded model (or None when running in demo mode)."""

    def __init__(self) -> None:
        self.tokenizer: Any = None
        self.model: Any = None
        self.labels: list[str] = ["Negative", "Positive"]
        self.mode: str = "demo"
        self.model_name: str = FALLBACK_MODEL_NAME
        self.num_parameters: Optional[int] = None


state = AppState()

# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def _normalize_labels(id2label: dict[Any, Any]) -> list[str]:
    """Map model id2label entries to clean "Positive"/"Negative" labels."""
    normalized: dict[int, str] = {}
    for key, value in id2label.items():
        idx = int(key)
        raw = str(value).strip().lower()
        if raw in {"0", "label_0", "neg", "negative"}:
            normalized[idx] = "Negative"
        elif raw in {"1", "label_1", "pos", "positive"}:
            normalized[idx] = "Positive"
        else:
            normalized[idx] = str(value).strip().capitalize()
    return [normalized[i] for i in sorted(normalized)]


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D logit vector."""
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / exps.sum()


# ---------------------------------------------------------------------------
# Demo-mode fallback (used only when the fine-tuned model is not loaded)
# ---------------------------------------------------------------------------

POSITIVE_WORDS = {
    "great", "excellent", "amazing", "wonderful", "fantastic", "brilliant",
    "perfect", "loved", "love", "best", "beautiful", "outstanding", "superb",
    "enjoyed", "enjoyable", "charming", "delightful", "compelling",
    "masterpiece", "gripping", "refreshing", "hilarious", "touching",
    "powerful", "stellar", "flawless", "recommend", "recommended", "gem",
    "memorable", "impressive", "solid", "good",
}
NEGATIVE_WORDS = {
    "terrible", "awful", "horrible", "bad", "worst", "boring", "waste",
    "disappointing", "disappointment", "poor", "dull", "predictable",
    "cliche", "cliched", "weak", "annoying", "hate", "hated", "mess",
    "painful", "unwatchable", "mediocre", "lackluster", "cringe",
    "nonsense", "laughable", "stupid", "trash", "garbage", "flop",
    "overrated", "flat", "slow",
}


def _demo_predict(review: str) -> dict[str, Any]:
    """Keyword-heuristic sentiment used when the fine-tuned weights are absent."""
    tokens = review.lower().split()
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    score = pos - neg
    sentiment = "Positive" if score >= 0 else "Negative"
    confidence = 0.5 + min(0.45, abs(score) * 0.09)
    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "probabilities": {
            "Negative": round(1 - confidence if sentiment == "Positive" else confidence, 4),
            "Positive": round(confidence if sentiment == "Positive" else 1 - confidence, 4),
        },
    }


# ---------------------------------------------------------------------------
# Model loading & inference
# ---------------------------------------------------------------------------


def _load_model() -> None:
    """Load the fine-tuned model from MODEL_DIR if the artifacts exist."""
    if not TRANSFORMERS_AVAILABLE:
        print("[EncoderX] transformers/torch unavailable — starting in demo mode.")
        return
    if not (MODEL_DIR / "config.json").exists():
        print(
            "[EncoderX] No fine-tuned model found at "
            f"{MODEL_DIR.resolve()} — starting in demo mode.\n"
            "           Place the checkpoint there (or set MODEL_DIR) and restart "
            "to enable real BERT inference."
        )
        return

    print(f"[EncoderX] Loading fine-tuned model from {MODEL_DIR.resolve()} …")
    state.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    state.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    state.model.to(DEVICE)
    state.model.eval()

    config = state.model.config
    id2label = getattr(config, "id2label", None)
    if id2label:
        state.labels = _normalize_labels(dict(id2label))
    state.model_name = str(getattr(config, "_name_or_path", MODEL_DIR.name))
    state.num_parameters = sum(p.numel() for p in state.model.parameters())
    state.mode = "model"
    print(
        f"[EncoderX] Model ready on {DEVICE} "
        f"({state.num_parameters:,} parameters, labels={state.labels})."
    )


def _model_predict(review: str) -> dict[str, Any]:
    """Run BERT inference and return sentiment + probabilities."""
    inputs = state.tokenizer(
        review,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        logits = state.model(**inputs).logits[0].float().cpu().numpy()

    if logits.ndim == 0 or logits.shape == (1,):
        # Single-logit (sigmoid) head.
        pos_prob = float(1.0 / (1.0 + np.exp(-float(logits))))
        probs = {"Negative": 1.0 - pos_prob, "Positive": pos_prob}
        labels = ["Negative", "Positive"]
    else:
        softmax = _softmax(logits)
        labels = state.labels if len(state.labels) == len(softmax) else [
            f"Class {i}" for i in range(len(softmax))
        ]
        probs = {label: float(p) for label, p in zip(labels, softmax)}

    sentiment = max(probs, key=probs.__getitem__)
    return {
        "sentiment": sentiment,
        "confidence": round(probs[sentiment], 4),
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
    }


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    review: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Movie review text to classify.",
    )


class PredictionResponse(BaseModel):
    sentiment: str = Field(..., description="Predicted sentiment label.")
    confidence: float = Field(..., description="Confidence in the prediction (0–1).")
    probabilities: dict[str, float] = Field(
        ..., description="Class probabilities from the model."
    )
    inference_time_ms: float = Field(..., description="Inference latency in ms.")
    mode: str = Field(..., description='"model" (fine-tuned BERT) or "demo" fallback.')


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str
    labels: list[str]
    max_length: int
    device: str
    mode: str
    num_parameters: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    mode: str
    device: str


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_model()
    yield


app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "mode": state.mode,
        "endpoints": "POST /predict · GET /model-info · GET /health",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", mode=state.mode, device=str(DEVICE))


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(request: PredictRequest) -> PredictionResponse:
    """Classify a movie review as Positive/Negative with a confidence score."""
    review = request.review.strip()
    if not review:
        raise HTTPException(status_code=400, detail="Review text must not be empty.")

    started = time.perf_counter()
    result = _model_predict(review) if state.mode == "model" else _demo_predict(review)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return PredictionResponse(
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        inference_time_ms=round(elapsed_ms, 2),
        mode=state.mode,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["meta"])
def model_info() -> ModelInfoResponse:
    """Report which model is currently serving predictions."""
    return ModelInfoResponse(
        model_name=state.model_name if state.mode == "model" else f"{FALLBACK_MODEL_NAME} (demo heuristic)",
        labels=state.labels,
        max_length=MAX_LENGTH,
        device=str(DEVICE),
        mode=state.mode,
        num_parameters=state.num_parameters,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
