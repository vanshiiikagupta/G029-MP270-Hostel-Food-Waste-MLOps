import math

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

valid_data = {
    "students_registered": 300,
    "attendance_pct": 85,
    "menu_rating": 4,
    "temp_c": 30,
    "cooked_kg": 150,
    "plate_size_cm": 25,
    "meal": "lunch",
    "hostel": "Girls Hostel"
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_invalid_cooked_kg():
    data = valid_data.copy()
    data["cooked_kg"] = 1200

    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_invalid_plate_size():
    data = valid_data.copy()
    data["plate_size_cm"] = 8

    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_valid_prediction():
    response = client.post("/predict", json=valid_data)

    assert response.status_code == 200

    result = response.json()
    assert "prediction" in result
    assert isinstance(result["prediction"], (int, float))
    assert math.isfinite(result["prediction"])