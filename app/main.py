
from datetime import date
from pathlib import Path

import json
import logging
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import PredictionInput
from src.mp270.preprocessing import prepare_features



class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "event": record.getMessage(),
            "prediction": getattr(record, "prediction", None),
            "model_version": "v2"
        })


logger = logging.getLogger("mp270.predictions")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())

logger.handlers.clear()
logger.addHandler(handler)
logger.propagate = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

REFERENCE_DATE = date(2026, 3, 30)

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Hostel Mess Food Waste Prediction API",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        input_data = data.model_dump()
        input_data["date"] = REFERENCE_DATE

        input_df = pd.DataFrame([input_data])
        features = prepare_features(input_df)

        prediction = float(model.predict(features)[0])

        logger.info("prediction_completed", extra={"prediction": prediction})
        return {"prediction": prediction}

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        ) from exc
