
from fastapi.testclient import TestClient
from src.utils.recommendation_cache import (
    clear_recommendation_cache,
)
from src.api.main import app
from src.api.main import app, artifacts
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["version"] == "1.0.0"


def test_valid_customer_recommendations():
    response = client.post(
        "/recommend",
        json={"user_id": "U00003", "limit": 5},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == "U00003"
    assert data["total_recommendations"] == 5
    assert len(data["recommendations"]) == 5


def test_non_consenting_customer_gets_no_recommendations():
    response = client.post(
        "/recommend",
        json={"user_id": "U00001", "limit": 5},
    )

    assert response.status_code == 200
    assert response.json()["total_recommendations"] == 0
    assert response.json()["recommendations"] == []


def test_unknown_customer_gets_no_recommendations():
    response = client.post(
        "/recommend",
        json={"user_id": "U99999", "limit": 5},
    )

    assert response.status_code == 200
    assert response.json()["total_recommendations"] == 0


def test_invalid_recommendation_limit():
    response = client.post(
        "/recommend",
        json={"user_id": "U00003", "limit": 100},
    )

    assert response.status_code == 422

def test_recommender_artifacts_loaded():
    assert artifacts is not None

    required_keys = {
        "popularity_ranking",
        "vectorizer",
        "item_matrix",
        "users",
        "items",
        "interaction_matrix",
    }

    assert required_keys.issubset(artifacts.keys())
    assert len(artifacts["users"]) == 850
    assert len(artifacts["items"]) == 50
    assert artifacts["interaction_matrix"].nnz == 12905



def test_recommendation_cache_hit_and_miss():
    clear_recommendation_cache()

    payload = {
        "user_id": "U00003",
        "limit": 5,
    }

    # First request should generate recommendations.
    first_response = client.post("/recommend", json=payload)

    assert first_response.status_code == 200
    assert first_response.json()["cache_hit"] is False

    # Second request should return cached recommendations.
    second_response = client.post("/recommend", json=payload)

    assert second_response.status_code == 200
    assert second_response.json()["cache_hit"] is True

    # Both responses should contain identical recommendations.
    assert (
        first_response.json()["recommendations"]
        == second_response.json()["recommendations"]
    )

    clear_recommendation_cache()


def test_api_rate_limit():
    rate_limit_client = TestClient(
        app,
        client=("127.0.0.2", 50000),
    )

    payload = {
        "user_id": "U00003",
        "limit": 5,
    }

    for _ in range(10):
        response = rate_limit_client.post(
            "/recommend",
            json=payload,
        )
        assert response.status_code == 200

    response = rate_limit_client.post(
        "/recommend",
        json=payload,
    )

    assert response.status_code == 429
