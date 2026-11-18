
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

EVENT_WEIGHTS = {
    "view": 1,
    "click": 2,
    "save": 3,
    "enroll": 5,
}


def build_popularity_ranking(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    interactions: pd.DataFrame,
) -> pd.DataFrame:

    # Only use interactions from users who granted consent.
    consented_users = set(
        customers.loc[customers["consent"], "user_id"]
    )

    eligible_interactions = interactions[
        interactions["user_id"].isin(consented_users)
    ].copy()

    # Convert interaction events into numerical weights.
    eligible_interactions["event_weight"] = (
        eligible_interactions["event_type"].map(EVENT_WEIGHTS)
    )

    # Calculate popularity for each product.
    scores = (
        eligible_interactions
        .groupby("item_id")
        .agg(
            popularity_score=("event_weight", "sum"),
            interaction_count=("event_type", "size"),
            unique_users=("user_id", "nunique"),
        )
        .reset_index()
    )

    # Include products with zero interactions.
    ranking = products.merge(
        scores,
        on="item_id",
        how="left",
    )

    ranking[
        ["popularity_score", "interaction_count", "unique_users"]
    ] = ranking[
        ["popularity_score", "interaction_count", "unique_users"]
    ].fillna(0)

    ranking = ranking.sort_values(
        ["popularity_score", "interaction_count", "item_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return ranking


def recommend_popular(
    user_id: str,
    customers: pd.DataFrame,
    ranking: pd.DataFrame,
    interactions: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    customer = customers[
        customers["user_id"] == user_id
    ]

    if customer.empty:
        return pd.DataFrame()

    # Consent is mandatory.
    if not bool(customer.iloc[0]["consent"]):
        return pd.DataFrame()

    # Exclude previously interacted products.
    seen_items = set(
        interactions.loc[
            interactions["user_id"] == user_id,
            "item_id",
        ]
    )

    recommendations = ranking[
        (ranking["available"] == True)
        & (~ranking["item_id"].isin(seen_items))
    ].head(k).copy()

    recommendations["recommendation_reason"] = (
        "Popular among consented users"
    )

    return recommendations.reset_index(drop=True)


def main():

    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    interactions = pd.read_csv(RAW / "interactions.csv")

    ranking = build_popularity_ranking(
        customers,
        products,
        interactions,
    )

    REPORTS.mkdir(exist_ok=True)

    ranking.to_csv(
        REPORTS / "popularity_ranking.csv",
        index=False,
    )

    print("\nTOP 10 POPULAR FINANCIAL PRODUCTS")
    print(
        ranking[
            ["item_id", "product_name", "category",
             "popularity_score", "interaction_count",
             "unique_users", "available"]
        ].head(10).to_string(index=False)
    )

    print("\nCUSTOMER RECOMMENDATIONS")

    user_id = "U00003"

    results = recommend_popular(
        user_id,
        customers,
        ranking,
        interactions,
        k=5,
    )

    if results.empty:
        print("No recommendations: consent missing or no eligible products.")
    else:
        print(
            results[
                ["item_id", "product_name", "category",
                 "popularity_score", "recommendation_reason"]
            ].to_string(index=False)
        )

    print("\nSaved: reports/popularity_ranking.csv")


if __name__ == "__main__":
    main()
