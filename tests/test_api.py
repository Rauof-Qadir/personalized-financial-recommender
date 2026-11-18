
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
