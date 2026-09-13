# EncoderX — BERT Sentiment Analysis API (Backend)

FastAPI service that serves a fine-tuned BERT model for movie-review sentiment
classification (Positive / Negative) with a confidence score.

## Endpoints

| Method | Path          | Description                                    |
| ------ | ------------- | ---------------------------------------------- |
| POST   | `/predict`    | Classify a review → sentiment + confidence     |
| GET    | `/model-info` | Model name, labels, device, serving mode       |
| GET    | `/health`     | Liveness check                                 |
| GET    | `/`           | Service metadata                               |
| GET    | `/docs`       | Interactive OpenAPI/Swagger UI                 |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Loading your fine-tuned modelThe fine-tuned checkpoint lives in `backend/model/` (this repo already includes
it: `config.json`, `model.safetensors`, tokenizer files). If you re-train and
save a new checkpoint with `model.save_pretrained(...)` /
`tokenizer.save_pretrained(...)`, either overwrite that directory or point
`MODEL_DIR` at the new one:

```
model/
├── config.json             # id2label: 0 = NEGATIVE, 1 = POSITIVE
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

When the weights are present, `/predict` runs real BERT inference
(`mode: "model"`). If they are missing, the API starts in a clearly labelled
**demo mode** (`mode: "demo"`) using a keyword heuristic, so the full-stack
flow can be demoed before the checkpoint is added.

## Environment variables

| Variable     | Default       | Purpose                                  |
| ------------ | ------------- | ---------------------------------------- |
| `MODEL_DIR`  | `./model`     | Directory with the fine-tuned artifacts |
| `MODEL_NAME` | `bert-base-uncased` | Display name in demo mode          |
| `MAX_LENGTH` | `256`         | Max tokenized sequence length            |

## Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "An absolutely wonderful film with a gripping story."}'
```

```json
{
  "sentiment": "Positive",
  "confidence": 0.9871,
  "probabilities": { "Negative": 0.0129, "Positive": 0.9871 },
  "inference_time_ms": 34.12,
  "mode": "model"
}
```

CORS is open (`allow_origins=["*"]`) for local development with the Next.js
frontend; restrict it before deploying.
