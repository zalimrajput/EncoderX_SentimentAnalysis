# 🎬 EncoderX Sentiment Analysis

**Fine-tuned BERT movie-review sentiment classifier with a full-stack web UI.**
Enter a movie review → a fine-tuned `bert-base-uncased` model classifies it as
**Positive** or **Negative** with a confidence score — served through a modern
Next.js dashboard backed by a FastAPI inference API.

| Accuracy | F1-Score | ROC-AUC |
| :------: | :------: | :-----: |
| **92.08%** | **92.14%** | **97.65%** |

*Evaluated on the held-out test split of the IMDB movie-review dataset.*

---

## 🏗️ Architecture

```
User Review → Next.js Frontend → FastAPI /predict → Fine-tuned BERT
            → Sentiment + Confidence → Frontend Result Card
```

- **Frontend** — Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS 4
- **Backend** — Python FastAPI + Uvicorn, PyTorch, Hugging Face Transformers
- **Model** — `bert-base-uncased` fine-tuned on IMDB (109.5M parameters),
  served entirely server-side; only plain text travels over the wire

## 📁 Project structure

```
├── backend/                  FastAPI inference service
│   ├── main.py               POST /predict · GET /model-info · GET /health
│   ├── requirements.txt
│   └── model/                Fine-tuned checkpoint metadata (config + tokenizer)
│       └── model.safetensors ⚠️ 437 MB — NOT in git (see "Get the model")
└── frontend/                 Next.js dashboard
    ├── app/                  page.tsx · layout.tsx · globals.css
    ├── components/           ResultCard (verdict, confidence bar, probabilities)
    └── lib/api.ts            Typed API client
```

## 🚀 Quick start

### 1 — Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows PowerShell: venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --port 8000
```

The API starts on **http://localhost:8000** — interactive docs at `/docs`.
If the weights are missing it runs in a clearly-labelled **demo mode**
(keyword heuristic) instead of crashing.

### 2 — Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**, type (or paste) a movie review, hit
**✨ Analyze Sentiment** (or press `Ctrl/⌘ + Enter`).

## 🧠 Get the model

The fine-tuned weights (`model.safetensors`, 437 MB) are excluded from git to
keep the repository light. Place the file at:

```
backend/model/model.safetensors
```

**Options:**
- Download from the release / drive link: _[add your link here]_
- Or fine-tune it yourself on the IMDB dataset with Hugging Face
  `Trainer` / `bert-base-uncased` and save with `model.save_pretrained("model")`.

The small metadata files (`config.json`, `tokenizer.json`,
`tokenizer_config.json`) **are** versioned, so once the weights file is in
place the model loads directly — no other configuration needed.

> Without the weights file, the backend still runs in demo mode so you can
> test the full front-end ↔ back-end flow.

## 🔌 API reference

Base URL: `http://localhost:8000`

| Method | Path          | Description                              |
| ------ | ------------- | ---------------------------------------- |
| POST   | `/predict`    | Classify a review → sentiment + confidence |
| GET    | `/model-info` | Model name, labels, device, serving mode |
| GET    | `/health`     | Liveness check                           |
| GET    | `/docs`       | Interactive Swagger UI                   |

**Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "An absolutely wonderful film with a gripping story."}'
```

```json
{
  "sentiment": "Positive",
  "confidence": 0.9989,
  "probabilities": { "Negative": 0.0011, "Positive": 0.9989 },
  "inference_time_ms": 112.06,
  "mode": "model"
}
```

## ✨ Features

- Large review textarea with live character counter (10,000 chars)
- One-click example reviews (positive / negative / mixed)
- Loading spinner + skeleton while BERT runs
- Result card: verdict, confidence %, animated confidence bar,
  per-class probabilities, inference latency
- Graceful error handling and clear/reset
- Demo-mode fallback when weights are absent
- Dark, responsive glassmorphism UI

## ⚙️ Configuration

| Variable               | Default                 | Purpose                       |
| ---------------------- | ----------------------- | ----------------------------- |
| `MODEL_DIR`            | `./model`               | Checkpoint directory (backend) |
| `MAX_LENGTH`           | `256`                   | Max tokenized sequence length |
| `NEXT_PUBLIC_API_URL`  | `http://localhost:8000` | API base URL (frontend)       |

## 🛠️ Tech stack

`Python` · `FastAPI` · `Uvicorn` · `PyTorch` · `Transformers` · `TypeScript` ·
`Next.js` · `React` · `Tailwind CSS`

---

*EncoderX internship submission — BERT fine-tuning & sentiment analysis.*
