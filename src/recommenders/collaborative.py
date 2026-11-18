
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

EVENT_WEIGHTS = {
    "view": 1,
    "click": 2,
    "save": 3,
    "enroll": 5,
}


def build_interaction_matrix(customers, products, interactions):
    """Build a sparse user-item matrix using consented customers only."""

    consented_users = customers.loc[
        customers["consent"].eq(True), "user_id"
    ].tolist()

    users = consented_users
    items = products["item_id"].tolist()

    user_positions = {
        user_id: position for position, user_id in enumerate(users)
    }
    item_positions = {
        item_id: position for position, item_id in enumerate(items)
    }

    data = interactions[
        interactions["user_id"].isin(user_positions)
        & interactions["item_id"].isin(item_positions)
    ].copy()

    data["weight"] = data["event_type"].map(EVENT_WEIGHTS)
    data = data.dropna(subset=["weight"])

    rows = data["user_id"].map(user_positions).to_numpy()
    cols = data["item_id"].map(item_positions).to_numpy()
    values = data["weight"].to_numpy(dtype=float)

    matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(users), len(items)),
        dtype=float,
    )

    return users, items, matrix


def recommend_collaborative(
    user_id,
    customers,
    products,
    users,
    items,
    matrix,
    k=10,
    n_neighbors=20,
):
    """Recommend unseen products using similar customers."""

    if k <= 0:
        raise ValueError("k must be greater than 0")

    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be greater than 0")

    customer = customers.loc[customers["user_id"] == user_id]

    if customer.empty or not bool(customer.iloc[0]["consent"]):
        return pd.DataFrame()

    if user_id not in users:
        return pd.DataFrame()

    user_position = users.index(user_id)
    user_vector = matrix[user_position]

    if user_vector.nnz == 0:
        return pd.DataFrame()

    similarities = cosine_similarity(
        user_vector,
        matrix,
        dense_output=False,
    )

    neighbor_positions = similarities.indices
    neighbor_scores = similarities.data

    neighbors = [
        (position, score)
        for position, score in zip(
            neighbor_positions, neighbor_scores
        )
        if position != user_position and score > 0
    ]

    neighbors.sort(key=lambda x: x[1], reverse=True)
    neighbors = neighbors[:n_neighbors]

    if not neighbors:
        return pd.DataFrame()

    neighbor_indices = [position for position, _ in neighbors]
    similarity_weights = np.array(
        [score for _, score in neighbors],
        dtype=float,
    )

    neighbor_matrix = matrix[neighbor_indices]

    weighted_scores = np.asarray(
        neighbor_matrix.T @ similarity_weights
    ).ravel()

    total_similarity = similarity_weights.sum()

    if total_similarity > 0:
        weighted_scores /= total_similarity

    seen_positions = set(user_vector.indices)

    results = pd.DataFrame({
        "item_id": items,
        "collaborative_score": weighted_scores,
    })

    results = results[
        ~results.index.isin(seen_positions)
    ]

    results = results.merge(products, on="item_id", how="inner")

    results = results[results["available"].eq(True)].copy()

    results = results.sort_values(
        ["collaborative_score", "item_id"],
        ascending=[False, True],
    ).head(k)

    results["recommendation_reason"] = (
        "Recommended based on similar customers' interactions"
    )

    return results.reset_index(drop=True)


def main():
    customers = pd.read_csv(RAW / "customers.csv")
    products = pd.read_csv(RAW / "products.csv")
    interactions = pd.read_csv(RAW / "interactions.csv")

    users, items, matrix = build_interaction_matrix(
        customers, products, interactions
    )

    user_id = "U00003"

    recommendations = recommend_collaborative(
        user_id=user_id,
        customers=customers,
        products=products,
        users=users,
        items=items,
        matrix=matrix,
        k=10,
        n_neighbors=20,
    )

    REPORTS.mkdir(exist_ok=True)

    output_path = REPORTS / "collaborative_recommendations.csv"
    recommendations.to_csv(output_path, index=False)

    print("\nCOLLABORATIVE FILTERING RECOMMENDATIONS")
    print(recommendations.to_string(index=False))
    print(f"\nTotal recommendations: {len(recommendations)}")
    print(f"Users in matrix: {len(users)}")
    print(f"Products in matrix: {len(items)}")
    print(f"Non-zero interactions: {matrix.nnz}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
