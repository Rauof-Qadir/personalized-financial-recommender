
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.utils.logger import get_logger

from src.api.schemas import RecommendationRequest
from src.recommenders.hybrid import recommend_hybrid
from src.utils.recommendation_cache import (
    get_cached_recommendations,
    cache_recommendations,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"

MODELS_DIR = ROOT_DIR / "models"
ARTIFACT_PATH = MODELS_DIR / "recommender_artifacts.joblib"


app = FastAPI(
    title="Personalized Financial Recommendation API",
    description="Hybrid recommendation engine for financial products.",
    version="1.0.0",
)

logger = get_logger("recommendation_api")

logger.info("Recommendation API initialized")
logger.info("Recommendation artifacts loaded successfully")

# Load datasets once when the API starts.
customers = pd.read_csv(RAW_DIR / "customers.csv")
products = pd.read_csv(RAW_DIR / "products.csv")
interactions = pd.read_csv(
    RAW_DIR / "interactions.csv",
    parse_dates=["timestamp"],
)
# Load precomputed recommendation components once at API startup.
artifacts = joblib.load(ARTIFACT_PATH)

@app.get("/")
def home():
    return {
        "message": "Financial Recommendation API is running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.post("/recommend")
def get_recommendations(request: RecommendationRequest):

    logger.info(
        "Recommendation request received | user_id=%s | limit=%s",
        request.user_id,
        request.limit,
    )

    try:
        # Check cache first.
        cached_results = get_cached_recommendations(
            request.user_id,
            request.limit,
        )

        if cached_results is not None:
            logger.info(
                "Cache hit | user_id=%s",
                request.user_id,
            )

            return {
                "user_id": request.user_id,
                "total_recommendations": len(cached_results),
                "recommendations": cached_results,
                "cache_hit": True,
            }

        logger.info(
            "Cache miss | generating recommendations | user_id=%s",
            request.user_id,
        )

        recommendations = recommend_hybrid(
            user_id=request.user_id,
            customers=customers,
            products=products,
            interactions=interactions,
            k=request.limit,
            artifacts=artifacts,
        )

        results = recommendations.to_dict(orient="records")

        cache_recommendations(
            request.user_id,
            request.limit,
            results,
        )

        logger.info(
            "Recommendations generated | user_id=%s | count=%s",
            request.user_id,
            len(results),
        )

        return {
            "user_id": request.user_id,
            "total_recommendations": len(results),
            "recommendations": results,
            "cache_hit": False,
        }

    except Exception:
        logger.exception(
            "Recommendation generation failed | user_id=%s",
            request.user_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to generate recommendations.",
        )
