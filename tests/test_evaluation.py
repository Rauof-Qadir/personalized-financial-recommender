
import unittest

import pandas as pd

from src.evaluation.evaluate import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    category_diversity,
    temporal_split,
)


class TestEvaluationMetrics(unittest.TestCase):

    def test_precision_at_k(self):
        recommended = ["P1", "P2", "P3", "P4"]
        relevant = {"P1", "P3"}

        self.assertAlmostEqual(
            precision_at_k(recommended, relevant, 4),
            0.5,
        )

    def test_recall_at_k(self):
        recommended = ["P1", "P2", "P3"]
        relevant = {"P1", "P3", "P5"}

        self.assertAlmostEqual(
            recall_at_k(recommended, relevant, 3),
            2 / 3,
        )

    def test_ndcg_at_k(self):
        recommended = ["P1", "P2", "P3"]
        relevant = {"P1", "P3"}

        score = ndcg_at_k(recommended, relevant, 3)

        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1)

    def test_ndcg_perfect_ranking(self):
        recommended = ["P1", "P2", "P3"]
        relevant = {"P1", "P2"}

        self.assertAlmostEqual(
            ndcg_at_k(recommended, relevant, 3),
            1.0,
        )

    def test_empty_relevant_items(self):
        self.assertEqual(
            recall_at_k(["P1", "P2"], set(), 2),
            0.0,
        )

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            precision_at_k(["P1"], {"P1"}, 0)

    def test_category_diversity(self):
        products = pd.DataFrame({
            "item_id": ["P1", "P2", "P3"],
            "category": ["savings", "investing", "savings"],
        })

        score = category_diversity(
            ["P1", "P2", "P3"],
            products,
        )

        self.assertAlmostEqual(score, 2 / 3)

    def test_temporal_split(self):
        interactions = pd.DataFrame({
            "user_id": ["U1"] * 5,
            "item_id": ["P1", "P2", "P3", "P4", "P5"],
            "event_type": ["view"] * 5,
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=5,
                freq="D",
            ),
        })

        train, test = temporal_split(interactions, 0.20)

        self.assertEqual(len(train), 4)
        self.assertEqual(len(test), 1)
        self.assertEqual(test.iloc[0]["item_id"], "P5")

    def test_invalid_test_fraction(self):
        interactions = pd.DataFrame({
            "user_id": ["U1"],
            "item_id": ["P1"],
            "event_type": ["view"],
            "timestamp": ["2026-01-01"],
        })

        with self.assertRaises(ValueError):
            temporal_split(interactions, 1.0)


if __name__ == "__main__":
    unittest.main()
