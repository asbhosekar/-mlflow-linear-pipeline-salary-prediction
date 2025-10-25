"""
train.py

Train a (linear / ridge / lasso) regression model, optionally run hyperparameter tuning,
and log parameters, metrics and model to MLflow. Also saves a local exported_model directory
so the model can be served by the API.

Usage examples:
  python train.py --data-path sample_input_for_regression.csv --target target --tune --model-type ridge --alpha 0.1 1 10
  python train.py --data-path sample_input_for_regression.csv --target target

"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from datetime import datetime, timezone

def parse_args():
    parser = argparse.ArgumentParser(description="Train regression model and log to MLflow")
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--target", type=str, default="", help="Target column name. If empty, last column is used.")
    parser.add_argument("--model-type", type=str, choices=["linear","ridge","lasso"], default="linear")
    parser.add_argument("--tune", action="store_true", help="Whether to run grid search tuning (only for ridge/lasso)")
    parser.add_argument("--alpha", type=float, nargs="*", default=[0.1,1.0,10.0], help="Alpha values for Ridge/Lasso tuning")
    parser.add_argument("--cv", type=int, default=5, help="CV folds for GridSearch")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiment-name", type=str, default="LinearRegression-Experiment")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None,
                        help="If provided, set MLflow tracking URI (e.g., http://127.0.0.1:5000). Otherwise defaults to local file-based mlruns.")
    parser.add_argument("--register-model", type=str, default=None, help="If using an MLflow Model Registry, provide a name to register model under")
    parser.add_argument("--autolog", action="store_true", help="Enable mlflow.sklearn.autolog()")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    if args.autolog:
        mlflow.sklearn.autolog()

    df = pd.read_csv(args.data_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    if args.target:
        target_col = args.target
    else:
        target_col = df.columns[-1]
        print(f"No target specified. Using last column as target: {target_col}")

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in data")

    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    # Identify numeric and categorical features
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")

    # Choose estimator
    if args.model_type == "linear":
        estimator = LinearRegression()
    elif args.model_type == "ridge":
        estimator = Ridge()
    elif args.model_type == "lasso":
        estimator = Lasso()
    else:
        raise ValueError("Unsupported model type")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])

    param_grid = {}
    if args.tune and args.model_type in ("ridge","lasso"):
        param_grid = {"estimator__alpha": args.alpha}

    #timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    
    with mlflow.start_run(run_name=f"train_{args.model_type}_{timestamp}") as run:
        run_id = run.info.run_id
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        if args.tune and param_grid:
            print("Starting grid search with params:", param_grid)
            search = GridSearchCV(pipeline, param_grid=param_grid, cv=args.cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            best_params = search.best_params_
            print("Best params:", best_params)
            mlflow.log_params({k: float(v) for k,v in best_params.items()})
            model_to_save = best
        else:
            pipeline.fit(X_train, y_train)
            model_to_save = pipeline

        # Evaluate on test set
        preds = model_to_save.predict(X_test)
        rmse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("r2", float(r2))

        # Log the sklearn pipeline as an MLflow model artifact
        mlflow.sklearn.log_model(model_to_save, name="model", registered_model_name=args.register_model if args.register_model else None)

        # Also save a local exported_model folder for easy serving / API loading
        export_dir = os.path.abspath("exported_model")
        if os.path.exists(export_dir):
            import shutil
            shutil.rmtree(export_dir)
        mlflow.sklearn.save_model(model_to_save, export_dir)
        print(f"Saved exported model to: {export_dir}")

        # Save run metadata to a small JSON file
        meta = {
            "run_id": run_id,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "params": { "model_type": args.model_type }
        }
        with open("run_metadata.json","w") as f:
            json.dump(meta, f, indent=2)

    print("Training complete. MLflow run id:", run_id)
    print("To view results: start `mlflow ui` or run an MLflow tracking server and open the UI.")

if __name__ == "__main__":
    main()
