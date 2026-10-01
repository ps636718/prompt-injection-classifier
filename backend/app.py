"""
FastAPI backend for Prompt Injection Classifier.
Provides inference API endpoints for detecting adversarial prompts.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path so src imports work reliably
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import existing preprocessing and inference logic directly without duplication
from src.predict import PromptClassifier

# Global classifier instance loaded on startup
classifier: Optional[PromptClassifier] = None


def resolve_artifact_path(env_var_name: str, relative_filename: str) -> Path:
    """Resolve model artifact path using env var or standard directory locations."""
    if os.getenv(env_var_name):
        return Path(os.environ[env_var_name])

    candidates = [
        BASE_DIR / "outputs" / relative_filename,
        Path(__file__).resolve().parent / "outputs" / relative_filename,
        Path("outputs") / relative_filename,
        Path("..") / "outputs" / relative_filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return candidates[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once on startup."""
    global classifier
    model_path = resolve_artifact_path("MODEL_PATH", "model.pkl")
    tfidf_path = resolve_artifact_path("TFIDF_PATH", "tfidf_vectorizer.pkl")

    print(f"Loading classifier artifacts...")
    print(f"  Model path: {model_path}")
    print(f"  TF-IDF path: {tfidf_path}")

    try:
        classifier = PromptClassifier(
            model_path=model_path,
            tfidf_path=tfidf_path,
        )
        print("Classifier loaded successfully.")
    except Exception as exc:
        print(f"Failed to load classifier on startup: {exc}")
        classifier = None

    yield
    print("Shutting down classifier service.")


app = FastAPI(
    title="Prompt Injection Classifier API",
    description="Real-time binary classification for detecting LLM prompt injections and jailbreaks.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Configuration ────────────────────────────────────────────────────────
frontend_url = os.getenv("FRONTEND_URL", "*")
if frontend_url and frontend_url.strip() != "*":
    allowed_origins = [origin.strip() for origin in frontend_url.split(",") if origin.strip()]
    allow_credentials = True
else:
    allowed_origins = ["*"]
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., description="Prompt text to be analyzed for injection attacks.")


class PredictResponse(BaseModel):
    label: str = Field(..., description="'MALICIOUS' or 'BENIGN'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "Prompt Injection Classifier API",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint indicating service and model readiness."""
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
    }


# ── Mount Frontend UI (for local testing & single-service preview) ────────────
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend_ui")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Classify an incoming prompt as MALICIOUS or BENIGN.
    Reuses existing feature engineering and ensemble pipeline from src/predict.py.
    """
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or currently initializing. Please try again shortly.",
        )

    try:
        prediction = classifier.predict(request.text)
        return {
            "label": prediction["verdict"],
            "confidence": float(prediction["confidence"]),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )
