
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

FEEDBACK_FILE = (
    ROOT_DIR / "reports" / "recommendation_feedback.csv"
)


def analyze_feedback(feedback_path=FEEDBACK_FILE):
    """Calculate customer feedback metrics."""

    feedback_path = Path(feedback_path)

    if not feedback_path.exists():
        return {
            "total_feedback": 0,
            "helpful_count": 0,
            "not_helpful_count": 0,
            "already_owned_count": 0,
            "helpful_rate": 0.0,
            "product_feedback": pd.DataFrame(),
        }

    df = pd.read_csv(feedback_path)

    if df.empty:
        return {
            "total_feedback": 0,
            "helpful_count": 0,
            "not_helpful_count": 0,
            "already_owned_count": 0,
            "helpful_rate": 0.0,
            "product_feedback": pd.DataFrame(),
        }

    counts = df["feedback"].value_counts()

    helpful = int(counts.get("Helpful", 0))
    not_helpful = int(counts.get("Not Helpful", 0))
    already_owned = int(
        counts.get("Already Own This Product", 0)
    )

    total = len(df)

    helpful_rate = (
        helpful / total * 100
        if total > 0
        else 0.0
    )

    product_feedback = (
        df.groupby(
            ["item_id", "product_name", "feedback"]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    return {
        "total_feedback": total,
        "helpful_count": helpful,
        "not_helpful_count": not_helpful,
        "already_owned_count": already_owned,
        "helpful_rate": round(helpful_rate, 2),
        "product_feedback": product_feedback,
    }
