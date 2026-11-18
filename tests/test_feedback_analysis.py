
import pandas as pd

from src.evaluation.feedback_analysis import analyze_feedback


def test_feedback_analysis(tmp_path):
    feedback_file = tmp_path / "feedback.csv"

    df = pd.DataFrame({
        "timestamp": [
            "2026-11-18T10:00:00",
            "2026-11-18T10:05:00",
            "2026-11-18T10:10:00",
            "2026-11-18T10:15:00",
        ],
        "user_id": ["U001", "U002", "U003", "U004"],
        "item_id": ["P001", "P001", "P002", "P003"],
        "product_name": [
            "Savings Plan",
            "Savings Plan",
            "Investment Guide",
            "Insurance Guide",
        ],
        "feedback": [
            "Helpful",
            "Helpful",
            "Not Helpful",
            "Already Own This Product",
        ],
    })

    df.to_csv(feedback_file, index=False)

    result = analyze_feedback(feedback_file)

    assert result["total_feedback"] == 4
    assert result["helpful_count"] == 2
    assert result["not_helpful_count"] == 1
    assert result["already_owned_count"] == 1
    assert result["helpful_rate"] == 50.0
    assert len(result["product_feedback"]) == 3


def test_missing_feedback_file(tmp_path):
    result = analyze_feedback(
        tmp_path / "missing.csv"
    )

    assert result["total_feedback"] == 0
    assert result["helpful_rate"] == 0.0
    assert result["product_feedback"].empty
