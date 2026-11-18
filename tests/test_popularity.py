
import unittest
import pandas as pd

from src.recommenders.popularity import (
    build_popularity_ranking,
    recommend_popular,
)


class TestPopularityRecommender(unittest.TestCase):

    def setUp(self):

        self.customers = pd.DataFrame({
            "user_id": ["U1", "U2"],
            "consent": [True, False],
        })

        self.products = pd.DataFrame({
            "item_id": ["P1", "P2", "P3"],
            "product_name": ["Savings", "Budget", "Credit"],
            "category": ["savings", "budgeting", "credit"],
            "available": [True, True, False],
        })

        self.interactions = pd.DataFrame({
            "user_id": ["U1", "U1", "U2"],
            "item_id": ["P1", "P1", "P2"],
            "event_type": ["view", "enroll", "enroll"],
        })

        self.ranking = build_popularity_ranking(
            self.customers,
            self.products,
            self.interactions,
        )

    def test_event_weights(self):
        score = self.ranking.set_index("item_id")
        self.assertEqual(score.loc["P1", "popularity_score"], 6)

    def test_non_consenting_interactions_excluded(self):
        score = self.ranking.set_index("item_id")
        self.assertEqual(score.loc["P2", "popularity_score"], 0)

    def test_unavailable_products_excluded(self):
        result = recommend_popular(
            "U1", self.customers, self.ranking,
            self.interactions, k=5
        )
        self.assertNotIn("P3", result["item_id"].tolist())

    def test_non_consenting_user_gets_no_recommendations(self):
        result = recommend_popular(
            "U2", self.customers, self.ranking,
            self.interactions, k=5
        )
        self.assertTrue(result.empty)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            recommend_popular(
                "U1", self.customers, self.ranking,
                self.interactions, k=0
            )


if __name__ == "__main__":
    unittest.main()
