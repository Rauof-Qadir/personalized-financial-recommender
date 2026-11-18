
import unittest

import pandas as pd

from src.recommenders.collaborative import (
    build_interaction_matrix,
    recommend_collaborative,
)


class TestCollaborativeRecommender(unittest.TestCase):

    def setUp(self):
        self.customers = pd.DataFrame({
            "user_id": ["U1", "U2", "U3", "U4"],
            "consent": [True, True, False, True],
        })

        self.products = pd.DataFrame({
            "item_id": ["P1", "P2", "P3", "P4"],
            "product_name": [
                "Savings Guide",
                "Investment Guide",
                "Credit Guide",
                "Insurance Guide",
            ],
            "category": [
                "savings",
                "investing",
                "credit",
                "insurance",
            ],
            "available": [True, True, True, False],
        })

        self.interactions = pd.DataFrame({
            "user_id": ["U1", "U2", "U2", "U3"],
            "item_id": ["P1", "P1", "P2", "P3"],
            "event_type": ["save", "save", "enroll", "click"],
        })

        (
            self.users,
            self.items,
            self.matrix,
        ) = build_interaction_matrix(
            self.customers,
            self.products,
            self.interactions,
        )

    def test_matrix_contains_only_consented_users(self):
        self.assertEqual(set(self.users), {"U1", "U2", "U4"})

    def test_matrix_uses_event_weights(self):
        user_position = self.users.index("U2")
        item_position = self.items.index("P2")

        self.assertEqual(
            self.matrix[user_position, item_position],
            5,
        )

    def test_similar_user_products_are_recommended(self):
        recommendations = recommend_collaborative(
            "U1",
            self.customers,
            self.products,
            self.users,
            self.items,
            self.matrix,
            k=10,
        )

        self.assertIn("P2", recommendations["item_id"].tolist())

    def test_previously_seen_products_are_excluded(self):
        recommendations = recommend_collaborative(
            "U1",
            self.customers,
            self.products,
            self.users,
            self.items,
            self.matrix,
        )

        self.assertNotIn("P1", recommendations["item_id"].tolist())

    def test_unavailable_products_are_excluded(self):
        recommendations = recommend_collaborative(
            "U1",
            self.customers,
            self.products,
            self.users,
            self.items,
            self.matrix,
        )

        self.assertNotIn("P4", recommendations["item_id"].tolist())

    def test_non_consenting_user_gets_no_recommendations(self):
        recommendations = recommend_collaborative(
            "U3",
            self.customers,
            self.products,
            self.users,
            self.items,
            self.matrix,
        )

        self.assertTrue(recommendations.empty)

    def test_cold_start_user_gets_no_recommendations(self):
        recommendations = recommend_collaborative(
            "U4",
            self.customers,
            self.products,
            self.users,
            self.items,
            self.matrix,
        )

        self.assertTrue(recommendations.empty)

    def test_invalid_k_raises_error(self):
        with self.assertRaises(ValueError):
            recommend_collaborative(
                "U1",
                self.customers,
                self.products,
                self.users,
                self.items,
                self.matrix,
                k=0,
            )

    def test_invalid_neighbor_count_raises_error(self):
        with self.assertRaises(ValueError):
            recommend_collaborative(
                "U1",
                self.customers,
                self.products,
                self.users,
                self.items,
                self.matrix,
                n_neighbors=0,
            )


if __name__ == "__main__":
    unittest.main()
