"""
predict_api/app.py

A small FastAPI app that loads the exported_model directory (created by train.py)
and exposes a /predict endpoint which accepts JSON arrays of records.

Run:
  export MODEL_PATH=./exported_model
  uvicorn predict_api.app:app --host 0.0.0.0 --port 8000

Example payload:
  {
    "records": [
      {"feature1": 1.2, "feature2": "A", "feature3": 10},
      {"feature1": 3.4, "feature2": "B", "feature3": 20}
    ]
  }

"""
import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.getcwd(), "exported_model"))

app = FastAPI(title="Linear Regression Model API")

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]

@app.on_event("startup")
def load_model():
    global model
    try:
        model = mlflow.sklearn.load_model(MODEL_PATH)
        print("Loaded model from:", MODEL_PATH)
    except Exception as e:
        print("Could not load model at startup:", e)
        model = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_path": MODEL_PATH}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")
    try:
        df = pd.DataFrame(req.records)
        # model expects the same preprocessing pipeline as training; exported_model includes the pipeline
        preds = model.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
