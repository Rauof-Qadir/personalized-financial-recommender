
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import RecommendationRequest
from src.recommenders.hybrid import recommend_hybrid


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"


app = FastAPI(
    title="Personalized Financial Recommendation API",
    description="Hybrid recommendation engine for financial products.",
    version="1.0.0",
)


# Load datasets once when the API starts.
customers = pd.read_csv(RAW_DIR / "customers.csv")
products = pd.read_csv(RAW_DIR / "products.csv")
interactions = pd.read_csv(
    RAW_DIR / "interactions.csv",
    parse_dates=["timestamp"],
)


@app.get("/")
def home():
    return {
        "message": "Financial Recommendation API is running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.post("/recommend")
def get_recommendations(request: RecommendationRequest):

    try:
        recommendations = recommend_hybrid(
            user_id=request.user_id,
            customers=customers,
            products=products,
            interactions=interactions,
            k=request.limit,
        )

        return {
            "user_id": request.user_id,
            "total_recommendations": len(recommendations),
            "recommendations": recommendations.to_dict(
                orient="records"
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate recommendations.",
        ) from exc
