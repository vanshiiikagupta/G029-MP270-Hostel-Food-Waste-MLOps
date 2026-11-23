from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

valid_data = {
    "students_registered": 200,
    "attendance_pct": 85,
    "menu_rating": 4,
    "temp_c": 30,
    "cooked_kg": 150,
    "plate_size_cm": 25,
    "meal": "lunch",
    "hostel": "A"
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


def test_model_unavailable():
    response = client.post("/predict", json=valid_data)

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Prediction model is not available yet."
    )