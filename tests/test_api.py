
from fastapi.testclient import TestClient

from src.mp270.api import app


client = TestClient(app)

VALID_PAYLOAD = {
    "students_registered": 500,
    "attendance_pct": 85.0,
    "menu_rating": 4.0,
    "temp_c": 30.0,
    "cooked_kg": 400.0,
    "plate_size_cm": 25.0,
    "meal": "lunch",
    "hostel": "Girls Hostel",
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_cooked_kg_validation():
    payload = VALID_PAYLOAD.copy()
    payload["cooked_kg"] = 1001

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_plate_size_validation():
    payload = VALID_PAYLOAD.copy()
    payload["plate_size_cm"] = 9

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

