from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


MODEL = joblib.load(Path(__file__).resolve().parent / "models" / "model.joblib")

app = FastAPI(title="Diabetes prediction")


class Patient(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: Patient) -> dict:
    data = pd.DataFrame([patient.model_dump()])
    return {
        "outcome": int(MODEL.predict(data)[0]),
        "probability": float(MODEL.predict_proba(data)[0, 1]),
    }
