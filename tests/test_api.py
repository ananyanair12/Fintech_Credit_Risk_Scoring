"""API smoke tests - need a trained model (python -m src.pipeline) and fastapi + httpx."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.config import MODEL_PATH  # noqa: E402

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet")

VALID = {
    "revolving_utilization": 0.35, "age": 42, "past_due_30_59": 0, "debt_ratio": 0.42,
    "monthly_income": 5400, "open_credit_lines": 8, "past_due_90": 0,
    "real_estate_loans": 1, "past_due_60_89": 0, "dependents": 2,
}


@pytest.fixture(scope="module")
def client():
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["model_loaded"] is True


def test_predict_returns_score_and_explanation(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["decision"] in {"approve", "decline"}
    assert len(body["shap_explanation"]["top_factors"]) > 0


def test_missing_income_is_allowed(client):
    payload = {**VALID, "monthly_income": None, "dependents": None}
    assert client.post("/predict", json=payload).status_code == 200


def test_invalid_input_rejected(client):
    assert client.post("/predict", json={**VALID, "age": 5}).status_code == 422
    assert client.post("/predict", json={**VALID, "revolving_utilization": -1}).status_code == 422
    bad = {k: v for k, v in VALID.items() if k != "debt_ratio"}
    assert client.post("/predict", json=bad).status_code == 422