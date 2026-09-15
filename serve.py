"""FastAPI service for the PGA Tour win predictor pipeline."""
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model

# Required for joblib.load to unpickle the fitted pipeline's custom step.
from pipeline_def import SeasonPercentileTransformer  # noqa: F401

ARTIFACT_PATH = os.environ.get(
    "PIPELINE_ARTIFACT_PATH", str(Path(__file__).parent / "pipeline.joblib")
)


def load_bundle(path: str):
    try:
        return joblib.load(path), None
    except Exception as exc:  # noqa: BLE001 -- any load failure should 503, not crash the process
        return None, str(exc)


_bundle, _load_error = load_bundle(ARTIFACT_PATH)


def _build_predict_model():
    if _bundle is None:
        # No real bounds available; keep the app importable so it still
        # serves a schema and every endpoint can 503 cleanly.
        return create_model("PredictRequest", season=(int, Field(..., ge=2000, le=2100)))
    fields = {
        col: (float, Field(..., ge=bounds["min"], le=bounds["max"]))
        for col, bounds in _bundle["feature_bounds"].items()
    }
    fields["season"] = (int, Field(..., ge=2000, le=2100))
    return create_model("PredictRequest", **fields)


PredictRequest = _build_predict_model()


class PredictResponse(BaseModel):
    predicted_win_season: bool
    win_probability: float


app = FastAPI(title="PGA Tour Win Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "info": "/info"}


def _require_bundle():
    if _bundle is None:
        raise HTTPException(status_code=503, detail=f"Model artifact not loaded: {_load_error}")
    return _bundle


@app.get("/info")
def info():
    bundle = _require_bundle()
    return {
        "status": "ok",
        "metadata": bundle["metadata"],
        "feature_bounds": bundle["feature_bounds"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    bundle = _require_bundle()
    row = request.model_dump()
    df = pd.DataFrame([row])
    proba = float(bundle["pipeline"].predict_proba(df)[0][1])
    return PredictResponse(predicted_win_season=proba >= 0.5, win_probability=proba)
