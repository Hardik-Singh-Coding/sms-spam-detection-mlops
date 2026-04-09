from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)


class PredictResponse(BaseModel):
    label: str
    probability: float


app = FastAPI(title="SMS Spam Shield", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest) -> PredictResponse:
    # Stubbed response for now
    text_lower = payload.text.lower()
    is_spammy = any(
        w in text_lower for w in ("win", "prize", "free", "call now", "urgent")
    )

    if is_spammy:
        return PredictResponse(label="spam", probability=0.9)
    return PredictResponse(label="ham", probability=0.9)
