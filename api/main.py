"""FastAPI scoring service.

Run locally:  uvicorn api.main:app --reload --port 8000
Docs:         http://localhost:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.predictor import Predictor
from api.schemas import ApplicantFeatures, ModelInfo, PredictResponse

logger = logging.getLogger("credit-risk-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = Predictor()
        logger.info("Model loaded: %s", app.state.predictor.bundle["model_name"])
    except FileNotFoundError:
        app.state.predictor = None
        logger.error("artifacts/model.joblib not found - run the training pipeline first.")
    yield


app = FastAPI(
    title="Credit Risk Scoring API",
    description="Predicts default risk and returns a per-applicant SHAP explanation.",
    version="1.0.0",
    lifespan=lifespan,
)


def _predictor() -> Predictor:
    p = app.state.predictor
    if p is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    return p


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.predictor is not None}


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    return _predictor().info()


@app.post("/predict", response_model=PredictResponse)
def predict(applicant: ApplicantFeatures):
    return _predictor().predict(applicant.model_dump())