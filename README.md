# MLflow Deployment Pipeline & API for Linear Regression

This project contains a ready-to-run MLflow-based training pipeline, a simple FastAPI-based model API, and documentation to run and fine-tune a linear regression model locally.

## Contents
- `train.py` - Train a Linear/Ridge/Lasso regression model, log to MLflow, and export the model to `exported_model/`.
- `hyperparameter_search.py` - Example randomized search to tune Ridge model and log best model to MLflow.
- `predict_api/` - FastAPI app and Dockerfile to serve `exported_model`.
- `MLproject`, `conda.yaml` - MLflow project metadata (optional).
- `requirements.txt` - Python dependencies for training.
- `predict_api/requirements.txt` - Dependencies for the API.
- `sample_input_for_regression.csv` - Example dataset (from the uploaded archive).

---

## Quick start (local)

### 1) Create a virtual environment and install dependencies
Using `venv`:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with conda:
```bash
conda env create -f conda.yaml
conda activate mlflow-linear-env
```

### 2) Start an MLflow tracking server (recommended)
This starts an MLflow server that stores metadata in a local SQLite DB and artifacts in `./mlruns/`.
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```
Then open MLflow UI at: http://127.0.0.1:5000

If you prefer not to run the server, the scripts will use the default local `mlruns/` folder.

### 3) Train the model and log to MLflow
Basic training:
```bash
python train.py --data-path sample_input_for_regression.csv --target <TARGET_COLUMN_NAME>

python train.py --data-path sample_input_for_regression.csv --target salary
```

Train with tuning (for Ridge/Lasso):
```bash
python train.py --data-path sample_input_for_regression.csv --target <TARGET_COLUMN_NAME> --model-type ridge --tune --alpha 0.001 0.01 0.1 1 10 100 --cv 5
```

This will:
- Create an MLflow run and log parameters, metrics and the fitted model.
- Save a local copy of the model to `exported_model/` for serving.

### 4) Serve the model (two options)

#### Option A — Use MLflow model server
```bash
mlflow models serve -m exported_model --no-conda -p 1234
```
Then POST JSON to `http://127.0.0.1:1234/invocations`.
```
{
    "dataframe_records": [
        {
            "id": 101,
            "age": 35,
            "years_experience": 10,
            "department": "Sales",
            "education": "Bachelors",
            "education_level": "Undergraduate",
            "performance_score": 4.2,
            "remote_work_pct": 20,
            "bonus": 5000,
            "hire_date": "2015-06-01",
            "certifications": 2
        }
    ]
}

```
#### Option B — Run the FastAPI app
Install API dependencies:
```bash
pip install -r predict_api/requirements.txt
```
Run:
```bash
#export MODEL_PATH=./exported_model
set MODEL_PATH=./exported_model
echo %MODEL_PATH% 
uvicorn predict_api.app:app --host 0.0.0.0 --port 8000
```
POST to `http://127.0.0.1:8000/predict` with payload:
```json
{
    "records": [
        {
            "id": 101,
            "age": 35,
            "years_experience": 10,
            "department": "Sales",
            "education": "Bachelors",
            "education_level": "Undergraduate",
            "performance_score": 4.2,
            "remote_work_pct": 20,
            "bonus": 5000,
            "hire_date": "2015-06-01",
            "certifications": 2
        }
    ]
}
```
Response:
```json
{
    "predictions": [
        75211.12961613911
    ]
}
```

---

## Fine-tuning / Hyperparameter search

There are two example approaches included:

1. **Grid search inside `train.py`** (use `--tune` with `--alpha ...`): uses `GridSearchCV` and logs the best params & model to MLflow.

2. **Randomized search example**: `python hyperparameter_search.py --data-path sample_input_for_regression.csv --target <TARGET> --n-iter 20`

### Tips for larger sweeps
- Use `RandomizedSearchCV` or `Optuna` for large parameter spaces.
- Run experiments in parallel (multiple machines or use MLflow to track).
- Use MLflow's `mlflow.sklearn.autolog()` for auto-logging during quick experiments.
- Consider saving intermediate artifacts (example predictions, feature importance).

---

## Notes & Next steps
- You can configure `MLFLOW_TRACKING_URI` env var or pass `--mlflow-tracking-uri` to `train.py` to point to a remote MLflow server.
- To register models in MLflow Model Registry, run a tracking server and pass `--register-model <ModelName>` to `train.py`.
- The exported model is an MLflow-compatible model — it can be served with `mlflow models serve` or loaded using `mlflow.sklearn.load_model`.

---
## Docker run

```
docker build -t ml-predict-api:latest -f predict_api/Dockerfile .


docker run --rm -p 9000:9000 `
  -e MODEL_PATH=/app/exported_model `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 `
  -v "${PWD}\exported_model:/app/exported_model:ro" `
  --name ml-predict-api ml-predict-api:latest

```
## Run model

```
http://127.0.0.1:9000/predict
```
```json
{
    "records": [
        {
            "id": 102,
            "age": 35,
            "years_experience": 10,
            "department": "Sales",
            "education": "Bachelors",
            "education_level": "Undergraduate",
            "performance_score": 5.2,
            "remote_work_pct": 20,
            "bonus": 5000,
            "hire_date": "2015-06-01",
            "certifications": 2
        }
    ]
}
```
## Output

```
{
    "predictions": [
        77211.19380808137
    ]
}
```

## Model versioning
- Run the model with different input data, experiment_name, parameter or model_type
- Register this model of new experiment with existing registered model. Version will change automatically

```
python train.py --data-path sample_input_for_regression.csv --target salary --experiment-name LinearRegression_V2

```
