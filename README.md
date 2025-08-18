# Iris Classification API

## Problem
Classify iris flowers (Setosa, Versicolor, Virginica) from 4 numeric features:
sepal_length, sepal_width, petal_length, petal_width.

## Model
- Algorithm: RandomForestClassifier
- Dataset: scikit-learn built-in Iris
- Metric: Accuracy on test split printed during training
- Files saved: `model.pkl`, `target_names.pkl`, `feature_names.pkl`

## Endpoints
- `GET /` — Health check
- `POST /predict` — Predict species, returns `prediction` and `confidence`
- `GET /model-info` — Model metadata (type, features, classes)

## Example Request
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
