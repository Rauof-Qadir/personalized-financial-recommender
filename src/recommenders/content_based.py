
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

EVENT_WEIGHTS = {
    "view": 1,
    "click": 2,
    "save": 3,
    "enroll": 5,
}


def build_item_matrix(products):
    """Convert product metadata into TF-IDF vectors."""

    product_text = (
        products["product_name"].fillna("") + " "
        + products["category"].fillna("") + " "
        + products["description"].fillna("") + " "
        + products["tags"].fillna("")
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    item_matrix = vectorizer.fit_transform(product_text)

    return vectorizer, item_matrix


def build_user_profile(
    user_id,
    customers,
    interactions,
    products,
    vectorizer,
    item_matrix,
):
    """Build a user preference vector from interests and behavior."""

    customer = customers.loc[
        customers["user_id"] == user_id
    ]

    if customer.empty or not bool(customer.iloc[0]["consent"]):
        return None

    interests = customer.iloc[0]["interests"].replace("|", " ")

    # Explicit interests support new users.
    interest_vector = vectorizer.transform([interests])

    user_history = interactions[
        interactions["user_id"] == user_id
    ].copy()

    if not user_history.empty:

        user_history["weight"] = (
            user_history["event_type"].map(EVENT_WEIGHTS)
        )

        product_positions = {
            item_id: position
            for position, item_id in enumerate(products["item_id"])
        }

        weighted_vectors = []

        for row in user_history.itertuples():
            position = product_positions.get(row.item_id)

            if position is not None:
                weighted_vectors.append(
                    (item_matrix[position], row.weight)
                )

        if weighted_vectors:
            history_vector = weighted_vectors[0][0] * weighted_vectors[0][1]

            for vector, weight in weighted_vectors[1:]:
                history_vector = history_vector + vector * weight

            history_vector = normalize(history_vector)

            profile = normalize(
                interest_vector + history_vector * 2
            )

            return profile

    # Cold-start fallback: use declared interests.
    return normalize(interest_vector)


def recommend_content_based(
    user_id,
    customers,
    products,
    interactions,
    vectorizer,
    item_matrix,
    k=10,
):
    """Return personalized, available, unseen recommendations."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    profile = build_user_profile(
        user_id,
        customers,
        interactions,
        products,
        vectorizer,
        item_matrix,
    )

    if profile is None:
        return pd.DataFrame()

    scores = cosine_similarity(
        profile,
        item_matrix,
    ).flatten()

    results = products.copy()

    results["similarity_score"] = scores

    seen_items = set(
        interactions.loc[
            interactions["user_id"] == user_id,
            "item_id",
        ]
    )

    results = results[
        results["available"].eq(True)
        & ~results["item_id"].isin(seen_items)
    ].copy()

    results = results.sort_values(
        ["similarity_score", "item_id"],
        ascending=[False, True],
    ).head(k)

    results["recommendation_reason"] = (
        "Matches customer interests and product content"
    )

    return results.reset_index(drop=True)


def main():

    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    interactions = pd.read_csv(RAW / "interactions.csv")

    vectorizer, item_matrix = build_item_matrix(products)

    user_id = "U00003"

    recommendations = recommend_content_based(
        user_id=user_id,
        customers=customers,
        products=products,
        interactions=interactions,
        vectorizer=vectorizer,
        item_matrix=item_matrix,
        k=10,
    )

    REPORTS.mkdir(exist_ok=True)

    output_path = REPORTS / "content_based_recommendations.csv"

    recommendations.to_csv(output_path, index=False)

    print("\nCONTENT-BASED RECOMMENDATIONS")

    if recommendations.empty:
        print("No recommendations available.")
    else:
        print(
            recommendations[
                [
                    "item_id",
                    "product_name",
                    "category",
                    "similarity_score",
                    "recommendation_reason",
                ]
            ].to_string(index=False)
        )

    print("\nTotal recommendations:", len(recommendations))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
