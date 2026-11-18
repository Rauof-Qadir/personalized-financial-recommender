
import unittest
import pandas as pd

from src.recommenders.content_based import (
    build_item_matrix,
    recommend_content_based,
)


class TestContentBasedRecommender(unittest.TestCase):

    def setUp(self):

        self.customers = pd.DataFrame({
            "user_id": ["U1", "U2"],
            "interests": ["savings|budgeting", "investing"],
            "consent": [True, False],
        })

        self.products = pd.DataFrame({
            "item_id": ["P1", "P2", "P3"],
            "product_name": [
                "Emergency Savings Guide",
                "Monthly Budget Planner",
                "Investment Fundamentals",
            ],
            "category": [
                "savings", "budgeting", "investing"
            ],
            "description": [
                "Learn emergency savings and saving money",
                "Learn personal budgeting and expenses",
                "Learn investment and portfolio fundamentals",
            ],
            "tags": [
                "savings|education",
                "budgeting|education",
                "investing|education",
            ],
            "available": [True, True, False],
        })

        self.interactions = pd.DataFrame({
            "user_id": ["U1"],
            "item_id": ["P1"],
            "event_type": ["save"],
        })

        self.vectorizer, self.item_matrix = build_item_matrix(
            self.products
        )

    def recommend(self, user_id, k=5):

        return recommend_content_based(
            user_id,
            self.customers,
            self.products,
            self.interactions,
            self.vectorizer,
            self.item_matrix,
            k,
        )

    def test_recommendations_are_personalized(self):
        result = self.recommend("U1")
        self.assertGreater(len(result), 0)

    def test_previously_seen_items_excluded(self):
        result = self.recommend("U1")
        self.assertNotIn("P1", result["item_id"].tolist())

    def test_unavailable_items_excluded(self):
        result = self.recommend("U1")
        self.assertNotIn("P3", result["item_id"].tolist())

    def test_non_consenting_user_gets_no_recommendations(self):
        result = self.recommend("U2")
        self.assertTrue(result.empty)

    def test_cold_start_user_uses_interests(self):
        self.interactions = self.interactions.iloc[0:0]
        result = self.recommend("U1")
        self.assertGreater(len(result), 0)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            self.recommend("U1", k=0)


if __name__ == "__main__":
    unittest.main()
