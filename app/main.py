from fastapi import FastAPI, HTTPException
from app.schemas import PredictionInput

app = FastAPI(
    title="Hostel Mess Food Waste Prediction API",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PredictionInput):
    raise HTTPException(
        status_code=503,
        detail="Prediction model is not available yet."
    )