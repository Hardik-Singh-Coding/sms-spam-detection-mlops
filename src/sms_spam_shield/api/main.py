from __future__ import annotations
import joblib
from pathlib import Path
from fastapi import HTTPException, FastAPI
from pydantic import BaseModel, Field

# Specifying the path
BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"

# Loading the model
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print(f"Warning: Model not found at {MODEL_PATH}")


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)


class PredictResponse(BaseModel):
    label: str
    probability: float


app = FastAPI(title="SMS Spam Shield", version="0.1.0")


# ENDPOINTS
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest) -> PredictResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact missing")

    prediction = model.predict([payload.text])[0]

    probabilities = model.predict_proba([payload.text])[0]

    label_map = {"ham": "ham", "spam": "spam"}

    return PredictResponse(
        label=label_map[prediction], probability=float(max(probabilities))
    )
