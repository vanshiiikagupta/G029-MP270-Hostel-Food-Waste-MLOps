
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.mp270.preprocessing import prepare_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "champion_model.joblib"

# Load the complete trained preprocessing + regression pipeline once.
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="MP270 Hostel Food Waste Prediction API",
    version="1.0",
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    students_registered: int
    attendance_pct: float
    menu_rating: float
    temp_c: float
    cooked_kg: float = Field(le=1000)
    plate_size_cm: float = Field(ge=10)
    meal: str
    hostel: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest):
    data = request.model_dump()

    # The client contract does not include a date.
    # Use the median training date as a fixed reference.
    data["date"] = date(2026, 4, 1)

    input_df = pd.DataFrame([data])
    features = prepare_features(input_df)

    prediction = float(model.predict(features)[0])

    return {"prediction": prediction}

