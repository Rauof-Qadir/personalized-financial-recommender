
from pathlib import Path

import pandas as pd

from src.recommenders.popularity import (
    build_popularity_ranking,
    recommend_popular,
)
from src.recommenders.content_based import (
    build_item_matrix,
    recommend_content_based,
)
from src.recommenders.collaborative import (
    build_interaction_matrix,
    recommend_collaborative,
)


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

DEFAULT_WEIGHTS = {
    "popularity_score": 0.25,
    "similarity_score": 0.35,
    "collaborative_score": 0.40,
}


def normalize_scores(series):
    """Scale recommendation scores between 0 and 1."""

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return series.apply(lambda value: 1.0 if value > 0 else 0.0)

    return (series - minimum) / (maximum - minimum)


def recommend_hybrid(
    user_id,
    customers,
    products,
    interactions,
    k=10,
    n_neighbors=20,
    weights=None,
):
    """Combine popularity, content, and collaborative recommendations."""

    if k <= 0:
        raise ValueError("k must be greater than 0")

    weights = weights or DEFAULT_WEIGHTS

    if not abs(sum(weights.values()) - 1.0) < 1e-9:
        raise ValueError("Recommendation weights must sum to 1")

    customer = customers.loc[customers["user_id"] == user_id]

    if customer.empty or not bool(customer.iloc[0]["consent"]):
        return pd.DataFrame()

    # 1. Popularity recommendations
    popularity_ranking = build_popularity_ranking(
        customers, products, interactions
    )

    popularity_results = recommend_popular(
        user_id,
        customers,
        popularity_ranking,
        interactions,
        k=len(products),
    )

    # 2. Content-based recommendations
    vectorizer, item_matrix = build_item_matrix(products)

    content_results = recommend_content_based(
        user_id,
        customers,
        products,
        interactions,
        vectorizer,
        item_matrix,
        k=len(products),
    )

    # 3. Collaborative recommendations
    users, items, interaction_matrix = build_interaction_matrix(
        customers, products, interactions
    )

    collaborative_results = recommend_collaborative(
        user_id,
        customers,
        products,
        users,
        items,
        interaction_matrix,
        k=len(products),
        n_neighbors=n_neighbors,
    )

    # Start with all products and attach each algorithm's scores.
    scores = products[["item_id"]].copy()

    score_sources = [
        (popularity_results, "popularity_score"),
        (content_results, "similarity_score"),
        (collaborative_results, "collaborative_score"),
    ]

    for result, score_column in score_sources:
        if result.empty or score_column not in result.columns:
            scores[score_column] = 0.0
        else:
            source = result[["item_id", score_column]].drop_duplicates(
                "item_id"
            )
            scores = scores.merge(source, on="item_id", how="left")
            scores[score_column] = scores[score_column].fillna(0.0)

    # Normalize scores so the three algorithms are comparable.
    for column in DEFAULT_WEIGHTS:
        scores[column] = normalize_scores(scores[column])

    # Calculate final weighted score.
    scores["hybrid_score"] = sum(
        scores[column] * weights[column]
        for column in weights
    )

    results = products.merge(scores, on="item_id", how="inner")

    seen_items = set(
        interactions.loc[
            interactions["user_id"] == user_id, "item_id"
        ]
    )

    results = results[
        results["available"].eq(True)
        & ~results["item_id"].isin(seen_items)
    ].copy()

    results = results.sort_values(
        ["hybrid_score", "item_id"],
        ascending=[False, True],
    ).head(k)

    def explain(row):
        signals = []

        if row["popularity_score"] > 0:
            signals.append("popular")

        if row["similarity_score"] > 0:
            signals.append("matches interests")

        if row["collaborative_score"] > 0:
            signals.append("liked by similar customers")

        return (
            "Recommended because it is " + ", ".join(signals)
            if signals
            else "Recommended by combined ranking"
        )

    results["recommendation_reason"] = results.apply(explain, axis=1)

    return results.reset_index(drop=True)


def main():
    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    interactions = pd.read_csv(RAW / "interactions.csv")

    user_id = "U00003"

    recommendations = recommend_hybrid(
        user_id,
        customers,
        products,
        interactions,
        k=10,
    )

    REPORTS.mkdir(exist_ok=True)

    output_path = REPORTS / "hybrid_recommendations.csv"
    recommendations.to_csv(output_path, index=False)

    print("\nHYBRID FINANCIAL PRODUCT RECOMMENDATIONS")
    print(recommendations.to_string(index=False))
    print(f"\nTotal recommendations: {len(recommendations)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
