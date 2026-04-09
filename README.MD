## Production-Ready SMS Spam Shield

A small MLOps-style project that serves an SMS spam classifier as a FastAPI service using a scikit-learn **Naive Bayes** model.

## Goals

- **Production-ready shape**: clean package structure, reproducible training, CI checks, and a stable API contract
- **Simple ML baseline**: strong classical NLP baseline with a scikit-learn Pipeline
- **Practical inference**: return both a **label** and a **probability**

## Design decisions (locked)

- **Model output**: returns `label` + `probability`
- **Training cadence**: training runs **locally only**; CI runs lint/format + tests only (no training)
- **CI tooling**: `ruff` (optionally `black` later)
- **Artifact format**: persisted `joblib` scikit-learn `Pipeline` (single file)

## API

### Predict

- **Endpoint**: `POST /predict`
- **Request body** (JSON):

```json
{
  "text": "Congratulations! You've won a prize. Call now..."
}
```

- **Response** (JSON):

```json
{
  "label": "spam",
  "probability": 0.97
}
```

#### Probability semantics

`probability` is the model confidence **for the returned `label`** as a value in \([0, 1]\).

## Local development (high level)

- Create and activate a virtual environment
- Install dependencies from `requirements.txt`
- Run the API server locally (command will be documented once the app entrypoint is added)

## Training & artifacts

- Training is executed **locally** and produces a single serialized artifact: a scikit-learn `Pipeline` saved via `joblib`.
- CI does **not** train the model; it only validates code quality and runs tests.

## Repository layout (planned)

- `src/sms_spam_shield/api/`: FastAPI app, routers, schemas
- `src/sms_spam_shield/ml/`: training/inference utilities, feature pipeline, artifact I/O
- `src/sms_spam_shield/core/`: shared settings, logging, constants
- `tests/`: unit/integration tests
