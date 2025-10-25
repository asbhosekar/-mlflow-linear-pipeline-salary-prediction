"""
hyperparameter_search.py

Example randomized hyperparameter search for Ridge model. This script demonstrates
how to run a search and log the best model to MLflow. It uses RandomizedSearchCV.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os, json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.data_path)
    target_col = args.target if args.target else df.columns[-1]
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state)

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", Ridge())])

    param_dist = {"estimator__alpha": np.logspace(-3, 3, 100)}
    search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=args.n_iter, cv=5, scoring="neg_root_mean_squared_error", random_state=args.random_state, n_jobs=-1)

    mlflow.set_experiment("Hyperparameter-Search")
    with mlflow.start_run(run_name="random_search_ridge"):
        search.fit(X_train, y_train)
        best = search.best_estimator_
        preds = best.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("r2", float(r2))
        mlflow.sklearn.log_model(best, "model")
        # Save exported copy
        if os.path.exists("exported_model"):
            import shutil
            shutil.rmtree("exported_model")
        mlflow.sklearn.save_model(best, "exported_model")
        print("Randomized search complete. Best score (RMSE):", rmse)

if __name__ == "__main__":
    main()
