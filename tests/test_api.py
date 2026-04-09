from fastapi.testclient import TestClient

from sms_spam_shield.api.main import app


def test_health() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_contract() -> None:
    client = TestClient(app)
    resp = client.post("/predict", json={"text": "hello"})
    assert resp.status_code == 200

    data = resp.json()
    assert set(data.keys()) == {"label", "probability"}
    assert data["label"] in {"spam", "ham"}
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0
