from pathlib import Path
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.api.schemas import RecommendationRequest
from src.recommenders.hybrid import recommend_hybrid
from src.utils.logger import get_logger
from src.utils.monitoring import record_api_metrics
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

limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)

app.add_middleware(SlowAPIMiddleware)

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
@limiter.limit("10/minute")
def get_recommendations(
    request: Request,
    recommendation_request: RecommendationRequest,
):
    start_time = time.perf_counter()
    status_code = 200
    cache_hit = False
    recommendation_count = 0

    user_id = recommendation_request.user_id
    limit = recommendation_request.limit

    logger.info(
        "Recommendation request received | user_id=%s | limit=%s",
        user_id,
        limit,
    )

    try:
        cached_results = get_cached_recommendations(user_id, limit)

        if cached_results is not None:
            cache_hit = True
            recommendation_count = len(cached_results)

            logger.info("Cache hit | user_id=%s", user_id)

            return {
                "user_id": user_id,
                "total_recommendations": recommendation_count,
                "recommendations": cached_results,
                "cache_hit": True,
            }

        logger.info("Cache miss | generating recommendations | user_id=%s", user_id)

        recommendations = recommend_hybrid(
            user_id=user_id,
            customers=customers,
            products=products,
            interactions=interactions,
            k=limit,
            artifacts=artifacts,
        )

        results = recommendations.to_dict(orient="records")
        recommendation_count = len(results)

        cache_recommendations(user_id, limit, results)

        logger.info(
            "Recommendations generated | user_id=%s | count=%s",
            user_id,
            recommendation_count,
        )

        return {
            "user_id": user_id,
            "total_recommendations": recommendation_count,
            "recommendations": results,
            "cache_hit": False,
        }

    except Exception:
        status_code = 500

        logger.exception(
            "Recommendation generation failed | user_id=%s",
            user_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to generate recommendations.",
        )

    finally:
        response_time_ms = (
            time.perf_counter() - start_time
        ) * 1000

        record_api_metrics(
            status_code=status_code,
            response_time_ms=response_time_ms,
            cache_hit=cache_hit,
            recommendation_count=recommendation_count,
        )