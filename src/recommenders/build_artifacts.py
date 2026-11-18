
import pandas as pd

from src.recommenders.hybrid import (
    build_recommender_artifacts,
    save_recommender_artifacts,
)


def main():
    customers = pd.read_csv("data/raw/customers.csv")
    products = pd.read_csv("data/raw/products.csv")
    interactions = pd.read_csv(
        "data/raw/interactions.csv",
        parse_dates=["timestamp"],
    )

    print("Building recommendation artifacts...")

    artifacts = build_recommender_artifacts(
        customers,
        products,
        interactions,
    )

    save_recommender_artifacts(artifacts)

    print("Artifact building completed.")
    print("Users:", len(artifacts["users"]))
    print("Products:", len(artifacts["items"]))
    print("Interactions:", artifacts["interaction_matrix"].nnz)


if __name__ == "__main__":
    main()
