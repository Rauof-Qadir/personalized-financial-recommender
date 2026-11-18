
import csv

import pytest

from src.utils import feedback


def test_save_recommendation_feedback(tmp_path, monkeypatch):

    feedback_file = tmp_path / "recommendation_feedback.csv"

    monkeypatch.setattr(
        feedback,
        "REPORTS_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        feedback,
        "FEEDBACK_FILE",
        feedback_file,
    )

    record = feedback.save_recommendation_feedback(
        user_id="U00003",
        item_id="P010",
        product_name="Understanding Diversification",
        feedback="Helpful",
    )

    assert record["user_id"] == "U00003"
    assert record["item_id"] == "P010"
    assert record["feedback"] == "Helpful"

    with feedback_file.open(
        newline="",
        encoding="utf-8",
    ) as file:

        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["feedback"] == "Helpful"


def test_invalid_feedback_is_rejected():

    with pytest.raises(ValueError):

        feedback.save_recommendation_feedback(
            user_id="U00003",
            item_id="P010",
            product_name="Test Product",
            feedback="Bad Rating",
        )


def test_missing_customer_information_is_rejected():

    with pytest.raises(ValueError):

        feedback.save_recommendation_feedback(
            user_id="",
            item_id="P010",
            product_name="Test Product",
            feedback="Helpful",
        )
