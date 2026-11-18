
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.recommenders.hybrid import recommend_hybrid


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

RELEVANT_EVENTS = {"click", "save", "enroll"}


def precision_at_k(recommended, relevant, k):
    if k <= 0:
        raise ValueError("k must be greater than 0")

    hits = len(set(recommended[:k]) & set(relevant))
    return hits / k


def recall_at_k(recommended, relevant, k):
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant:
        return 0.0

    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant = set(relevant)

    if not relevant:
        return 0.0

    dcg = sum(
        1 / np.log2(position + 2)
        for position, item in enumerate(recommended[:k])
        if item in relevant
    )

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1 / np.log2(position + 2)
        for position in range(ideal_hits)
    )

    return dcg / idcg if idcg else 0.0


def category_diversity(recommended, products):
    """Calculate pairwise category diversity within recommendations."""

    if len(recommended) < 2:
        return 0.0

    category_lookup = products.set_index("item_id")["category"].to_dict()

    categories = [
        category_lookup[item]
        for item in recommended
        if item in category_lookup
    ]

    if len(categories) < 2:
        return 0.0

    total_pairs = 0
    different_pairs = 0

    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            total_pairs += 1

            if categories[i] != categories[j]:
                different_pairs += 1

    return different_pairs / total_pairs


def temporal_split(interactions, test_fraction=0.20):
    """Split each user's interactions chronologically."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    train_parts = []
    test_parts = []

    ordered = interactions.sort_values(
        ["user_id", "timestamp"]
    )

    for _, group in ordered.groupby("user_id", sort=True):
        group = group.reset_index(drop=True)

        if len(group) < 2:
            train_parts.append(group)
            continue

        test_count = max(1, int(np.ceil(len(group) * test_fraction)))
        split_position = len(group) - test_count

        train_parts.append(group.iloc[:split_position])
        test_parts.append(group.iloc[split_position:])

    train = (
        pd.concat(train_parts, ignore_index=True)
        if train_parts else interactions.iloc[0:0].copy()
    )

    test = (
        pd.concat(test_parts, ignore_index=True)
        if test_parts else interactions.iloc[0:0].copy()
    )

    return train, test


def evaluate_hybrid(
    customers,
    products,
    interactions,
    k=10,
    max_users=100,
    test_fraction=0.20,
):
    """Evaluate hybrid recommendations using chronological holdout data."""

    if k <= 0:
        raise ValueError("k must be greater than 0")

    if max_users <= 0:
        raise ValueError("max_users must be greater than 0")

    train, test = temporal_split(interactions, test_fraction)

    consented_users = set(
        customers.loc[customers["consent"].eq(True), "user_id"]
    )

    available_items = set(
        products.loc[products["available"].eq(True), "item_id"]
    )

    user_cases = []

    for user_id, user_test in test.groupby("user_id"):
        if user_id not in consented_users:
            continue

        train_seen = set(
            train.loc[train["user_id"].eq(user_id), "item_id"]
        )

        relevant = set(
            user_test.loc[
                user_test["event_type"].isin(RELEVANT_EVENTS),
                "item_id",
            ]
        )

        # Evaluate only unseen, available products.
        relevant = relevant - train_seen
        relevant = relevant & available_items

        if relevant:
            user_cases.append((user_id, relevant))

    if not user_cases:
        raise ValueError("No eligible users with relevant test interactions")

    # Deterministic sample keeps evaluation lightweight.
    rng = np.random.default_rng(42)

    if len(user_cases) > max_users:
        selected = rng.choice(
            len(user_cases),
            size=max_users,
            replace=False,
        )
        user_cases = [user_cases[i] for i in sorted(selected)]

    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    diversity_scores = []
    all_recommended = set()

    for user_id, relevant in user_cases:
        recommendations = recommend_hybrid(
            user_id=user_id,
            customers=customers,
            products=products,
            interactions=train,
            k=k,
        )

        recommended = (
            recommendations["item_id"].tolist()
            if not recommendations.empty
            else []
        )

        precision_scores.append(
            precision_at_k(recommended, relevant, k)
        )

        recall_scores.append(
            recall_at_k(recommended, relevant, k)
        )

        ndcg_scores.append(
            ndcg_at_k(recommended, relevant, k)
        )

        diversity_scores.append(
            category_diversity(recommended, products)
        )

        all_recommended.update(recommended)

    catalog_size = len(available_items)

    results = {
        "evaluation_type": "Temporal Holdout",
        "training_fraction": round(1 - test_fraction, 2),
        "testing_fraction": round(test_fraction, 2),
        "users_evaluated": len(user_cases),
        "k": k,
        "precision_at_k": float(np.mean(precision_scores)),
        "recall_at_k": float(np.mean(recall_scores)),
        "ndcg_at_k": float(np.mean(ndcg_scores)),
        "catalog_coverage": (
            len(all_recommended) / catalog_size
            if catalog_size else 0.0
        ),
        "unique_products_recommended": len(all_recommended),
        "available_catalog_size": catalog_size,
        "category_diversity": float(np.mean(diversity_scores)),
    }

    return results


def main():
    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    interactions = pd.read_csv(RAW / "interactions.csv")

    results = evaluate_hybrid(
        customers,
        products,
        interactions,
        k=10,
        max_users=100,
        test_fraction=0.20,
    )

    REPORTS.mkdir(exist_ok=True)

    json_path = REPORTS / "hybrid_evaluation.json"
    csv_path = REPORTS / "hybrid_evaluation.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    pd.DataFrame([results]).to_csv(csv_path, index=False)

    print("\nHYBRID RECOMMENDER EVALUATION")
    print("-" * 45)

    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
