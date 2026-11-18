
import csv
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

REPORTS_DIR = ROOT_DIR / "reports"
FEEDBACK_FILE = REPORTS_DIR / "recommendation_feedback.csv"

FIELDNAMES = [
    "timestamp",
    "user_id",
    "item_id",
    "product_name",
    "feedback",
]

ALLOWED_FEEDBACK = {
    "Helpful",
    "Not Helpful",
    "Already Own This Product",
}


def save_recommendation_feedback(
    user_id,
    item_id,
    product_name,
    feedback,
):
    """Save customer feedback for a recommended financial product."""

    if feedback not in ALLOWED_FEEDBACK:
        raise ValueError("Invalid recommendation feedback.")

    if not user_id or not item_id or not product_name:
        raise ValueError("Customer and product information is required.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    file_exists = FEEDBACK_FILE.exists()

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "item_id": item_id,
        "product_name": product_name,
        "feedback": feedback,
    }

    with FEEDBACK_FILE.open(
        mode="a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)

    return record
