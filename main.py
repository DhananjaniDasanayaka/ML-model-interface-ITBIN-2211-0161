# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

# ---- Load model and metadata on startup (module import is fine for this scope)
model = joblib.load("model.pkl")
target_names = joblib.load("target_names.pkl")
feature_names = joblib.load("feature_names.pkl")

app = FastAPI(
    title="Iris Classifier API",
    description="Predict iris species (setosa, versicolor, virginica) from 4 features.",
    version="1.0.0",
)

# ---- Input/Output Schemas
class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float | None = None

# ---- Endpoints
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Iris Classifier API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: IrisInput):
    """
    Accepts 4 numeric features and returns predicted species + confidence.
    """
    try:
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]])
        pred_idx = int(model.predict(features)[0])
        proba = float(model.predict_proba(features).max())

        return PredictionOutput(
            prediction=str(target_names[pred_idx]),
            confidence=round(proba, 3),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "problem_type": "classification",
        "features": list(feature_names),
        "classes": list(map(str, target_names)),
    }
