
import unittest

import pandas as pd

from src.recommenders.hybrid import recommend_hybrid


class TestHybridRecommender(unittest.TestCase):

    def setUp(self):
        self.customers = pd.DataFrame({
            "user_id": ["U1", "U2", "U3"],
            "interests": [
                "savings budgeting",
                "savings budgeting",
                "investing",
            ],
            "consent": [True, True, False],
        })

        self.products = pd.DataFrame({
            "item_id": ["P1", "P2", "P3", "P4"],
            "product_name": [
                "Emergency Savings Guide",
                "Monthly Budget Planner",
                "Investment Fundamentals",
                "Unavailable Savings Product",
            ],
            "category": [
                "savings",
                "budgeting",
                "investing",
                "savings",
            ],
            "description": [
                "Learn about emergency savings and saving money.",
                "Learn how to manage monthly budgets and expenses.",
                "Learn investment basics and portfolio management.",
                "Learn about savings accounts.",
            ],
            "tags": [
                "savings|emergency",
                "budgeting|expenses",
                "investing|portfolio",
                "savings",
            ],
            "available": [True, True, True, False],
            "minimum_age": [18, 18, 18, 18],
        })

        self.interactions = pd.DataFrame({
            "user_id": ["U1", "U2", "U2", "U3"],
            "item_id": ["P1", "P1", "P2", "P3"],
            "event_type": ["save", "save", "enroll", "click"],
        })

    def test_hybrid_returns_recommendations(self):
        results = recommend_hybrid(
            "U1",
            self.customers,
            self.products,
            self.interactions,
            k=10,
        )

        self.assertFalse(results.empty)
        self.assertIn("hybrid_score", results.columns)
        self.assertIn("recommendation_reason", results.columns)

    def test_seen_items_are_excluded(self):
        results = recommend_hybrid(
            "U1",
            self.customers,
            self.products,
            self.interactions,
        )

        self.assertNotIn("P1", results["item_id"].tolist())

    def test_unavailable_items_are_excluded(self):
        results = recommend_hybrid(
            "U1",
            self.customers,
            self.products,
            self.interactions,
        )

        self.assertNotIn("P4", results["item_id"].tolist())

    def test_non_consenting_user_gets_no_recommendations(self):
        results = recommend_hybrid(
            "U3",
            self.customers,
            self.products,
            self.interactions,
        )

        self.assertTrue(results.empty)

    def test_invalid_k_raises_error(self):
        with self.assertRaises(ValueError):
            recommend_hybrid(
                "U1",
                self.customers,
                self.products,
                self.interactions,
                k=0,
            )

    def test_weights_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            recommend_hybrid(
                "U1",
                self.customers,
                self.products,
                self.interactions,
                weights={
                    "popularity_score": 0.5,
                    "similarity_score": 0.5,
                    "collaborative_score": 0.5,
                },
            )

    def test_recommendations_are_sorted(self):
        results = recommend_hybrid(
            "U1",
            self.customers,
            self.products,
            self.interactions,
        )

        scores = results["hybrid_score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
