#!/usr/bin/env python3
"""Compare the pygamlab regression workflow with scikit-learn.

The workflow is the chemical-congress solution: two IQR filters, abs(log)
of solubility, then regression_PIPE with a MinMaxScaler and GridSearchCV.
The scikit-learn side builds the same Pipeline and the same parameter grid.

Metrics are MAE, RMSE, and R² on out-of-fold predictions.
Performance is GridSearch fit time and a separate predict time.
"""

from __future__ import annotations

import csv
import io
import json
import time
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygamlab
from pygamlab.ai_core import regression_PIPE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

plt.show = lambda *args, **kwargs: plt.close("all")
warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).resolve().parent / "outputs"
DATA_PATH = Path(__file__).resolve().parent / "chemical_congress.xlsx"
CV = 5
FEATURES = ["Mw", "Melting point", "density", "HBD", "HBA"]
TARGET = "solubility"

KNN_PARAMS = {
    "model__n_neighbors": [3, 5, 7, 9, 11],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],
    "model__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "model__leaf_size": [20, 30, 40, 50],
}
RF_PARAMS = {
    "model__n_estimators": [50, 100],
    "model__criterion": ["squared_error", "absolute_error", "friedman_mse"],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"],
    "model__bootstrap": [True, False],
}
LINEAR_PARAMS = {}
DT_PARAMS = {
    "model__criterion": ["squared_error", "absolute_error", "friedman_mse"],
    "model__splitter": ["best", "random"],
    "model__max_depth": [None, 5, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": [None, "sqrt", "log2"],
}
SVR_PARAMS = {
    "model__kernel": ["linear", "rbf", "poly", "sigmoid"],
    "model__C": [0.01, 0.1, 1, 10, 50, 100],
    "model__epsilon": [0.001, 0.01, 0.05, 0.1],
    "model__degree": [2, 3],
    "model__coef0": [-0.5, 0, 0.5],
}
MLP_PARAMS = {
    "model__hidden_layer_sizes": [(50,), (100, 100), (9, 6, 3), (9, 9, 9), (9, 6)],
    "model__activation": ["relu"],
    "model__solver": ["adam", "lbfgs"],
    "model__alpha": [0.0001, 0.001, 0.01, 0.1],
    "model__learning_rate": ["constant", "adaptive"],
    "model__max_iter": [500, 10000],
}


def iqr_mask(y):
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (y >= lower) & (y <= upper)


def load_congress():
    """Same preparation as ai_training_regression_congress.py."""
    frame = pd.read_excel(DATA_PATH)
    x = frame[FEATURES].to_numpy(float)
    y = frame[TARGET].to_numpy(float).reshape(-1, 1)
    for _ in range(2):
        mask = iqr_mask(y)
        x = x[mask.flatten()]
        y = y[mask]
    y = np.abs(np.log(y)).ravel()
    return x, y


def scores(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return mae, rmse, r2


def sklearn_pipeline(estimator, polynomial):
    steps = [("scaler", MinMaxScaler())]
    if polynomial:
        steps.append(("poly", PolynomialFeatures()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def fit_pygamlab(x, y, estimator, params, polynomial):
    workflow = regression_PIPE(x, y)
    workflow.make_pipe(
        model=estimator,
        poly=True if polynomial else None,
        specific_preprocessor=MinMaxScaler(),
    )
    start = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        workflow.fitting(myparams=params, cv=CV)
    fit_ms = (time.perf_counter() - start) * 1000.0
    mae, rmse, r2 = scores(workflow.y, workflow.ypred)
    return workflow, fit_ms, mae, rmse, r2


def fit_sklearn(x, y, estimator, params, polynomial):
    search = GridSearchCV(
        estimator=sklearn_pipeline(estimator, polynomial),
        param_grid=params,
        cv=CV,
        scoring="neg_mean_absolute_percentage_error",
        return_train_score=True,
        n_jobs=-1,
    )
    start = time.perf_counter()
    search.fit(x, y)
    prediction = cross_val_predict(search.best_estimator_, x, y, cv=CV)
    fit_ms = (time.perf_counter() - start) * 1000.0
    mae, rmse, r2 = scores(y, prediction)
    return search.best_estimator_, fit_ms, mae, rmse, r2


def inference_ms(predict, x, repeats=20):
    predict(x)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict(x)
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples))


def run_model(name, estimator_factory, params, polynomial, reference, x, y):
    print(f"running {name}")
    gam, gam_fit, gam_mae, gam_rmse, gam_r2 = fit_pygamlab(
        x, y, estimator_factory(), params, polynomial
    )
    sk_model, sk_fit, sk_mae, sk_rmse, sk_r2 = fit_sklearn(
        x, y, estimator_factory(), params, polynomial
    )
    gam_infer = inference_ms(gam.predicting, x)
    sk_infer = inference_ms(sk_model.predict, x)
    row = {
        "task": name,
        "reference": reference,
        "n_samples": int(len(y)),
        "n_features": int(x.shape[1]),
        "mae_pygamlab": round(gam_mae, 6),
        "mae_sklearn": round(sk_mae, 6),
        "mae_abs_error": round(abs(gam_mae - sk_mae), 6),
        "rmse_pygamlab": round(gam_rmse, 6),
        "rmse_sklearn": round(sk_rmse, 6),
        "rmse_abs_error": round(abs(gam_rmse - sk_rmse), 6),
        "r2_pygamlab": round(gam_r2, 6),
        "r2_sklearn": round(sk_r2, 6),
        "r2_abs_error": round(abs(gam_r2 - sk_r2), 6),
        "fit_time_pygamlab_ms": round(gam_fit, 3),
        "fit_time_sklearn_ms": round(sk_fit, 3),
        "inference_time_pygamlab_ms": round(gam_infer, 3),
        "inference_time_sklearn_ms": round(sk_infer, 3),
    }
    print(
        f"  MAE {row['mae_pygamlab']} / {row['mae_sklearn']}  "
        f"RMSE {row['rmse_pygamlab']} / {row['rmse_sklearn']}  "
        f"R2 {row['r2_pygamlab']} / {row['r2_sklearn']}  "
        f"fit {row['fit_time_pygamlab_ms']} / {row['fit_time_sklearn_ms']} ms"
    )
    return row


def cases():
    return [
        (
            "knn",
            lambda: KNeighborsRegressor(),
            KNN_PARAMS,
            False,
            "regression_PIPE(X, y).make_pipe(model=KNeighborsRegressor(), specific_preprocessor=MinMaxScaler()); fitting(myparams=knn_params)",
        ),
        (
            "random_forest",
            lambda: RandomForestRegressor(random_state=42),
            RF_PARAMS,
            False,
            "regression_PIPE(X, y).make_pipe(model=RandomForestRegressor(random_state=42), specific_preprocessor=MinMaxScaler()); fitting(myparams=rf_params)",
        ),
        (
            "linear_regression",
            lambda: LinearRegression(),
            LINEAR_PARAMS,
            False,
            "regression_PIPE(X, y).make_pipe(model=LinearRegression(), specific_preprocessor=MinMaxScaler()); fitting(myparams={})",
        ),
        (
            "decision_tree",
            lambda: DecisionTreeRegressor(random_state=42),
            DT_PARAMS,
            False,
            "regression_PIPE(X, y).make_pipe(model=DecisionTreeRegressor(random_state=42), specific_preprocessor=MinMaxScaler()); fitting(myparams=dt_params)",
        ),
        (
            "svr",
            lambda: SVR(),
            SVR_PARAMS,
            False,
            "regression_PIPE(X, y).make_pipe(model=SVR(), specific_preprocessor=MinMaxScaler()); fitting(myparams=svr_params)",
        ),
        (
            "mlp",
            lambda: MLPRegressor(random_state=42),
            MLP_PARAMS,
            True,
            "regression_PIPE(X, y).make_pipe(model=MLPRegressor(random_state=42), poly=True, specific_preprocessor=MinMaxScaler()); fitting(myparams=mlp_params)",
        ),
    ]


def write_csv(rows):
    path = OUT_DIR / "results.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_summary(rows):
    path = OUT_DIR / "summary.txt"
    header = (
        f"{'task':<20} {'n':>5} {'MAE_gam':>10} {'MAE_sk':>10} "
        f"{'RMSE_gam':>10} {'RMSE_sk':>10} {'R2_gam':>8} {'R2_sk':>8} "
        f"{'fit_gam':>10} {'fit_sk':>10} {'inf_gam':>8} {'inf_sk':>8}  reference"
    )
    lines = [
        "pygamlab regression_PIPE vs scikit-learn",
        f"pygamlab {pygamlab.__version__}",
        "",
        "Data: chemical_congress.xlsx, features Mw, Melting point, density, HBD, HBA.",
        "Target: abs(log(solubility)) after two IQR filters, as in the congress solution.",
        "Metrics are out-of-fold MAE, RMSE, and R². Times are milliseconds.",
        "fit is GridSearchCV. inference is predict on the full feature matrix.",
        "",
        header,
        "-" * len(header),
    ]
    for row in rows:
        lines.append(
            f"{row['task']:<20} {row['n_samples']:5d} "
            f"{row['mae_pygamlab']:10.4f} {row['mae_sklearn']:10.4f} "
            f"{row['rmse_pygamlab']:10.4f} {row['rmse_sklearn']:10.4f} "
            f"{row['r2_pygamlab']:8.4f} {row['r2_sklearn']:8.4f} "
            f"{row['fit_time_pygamlab_ms']:10.1f} {row['fit_time_sklearn_ms']:10.1f} "
            f"{row['inference_time_pygamlab_ms']:8.3f} {row['inference_time_sklearn_ms']:8.3f}  "
            f"{row['reference']}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y = load_congress()
    print(f"congress data: {len(y)} samples, {x.shape[1]} features")
    rows = []
    for name, factory, params, polynomial, reference in cases():
        rows.append(run_model(name, factory, params, polynomial, reference, x, y))
    payload = {
        "pygamlab_version": pygamlab.__version__,
        "dataset": str(DATA_PATH),
        "target": "abs(log(solubility))",
        "metrics": ["MAE", "RMSE", "R2"],
        "performance_metrics": ["fit_time_ms", "inference_time_ms"],
        "results": rows,
    }
    json_path = OUT_DIR / "results.json"
    json_path.write_text(json.dumps(payload, indent=2))
    csv_path = write_csv(rows)
    summary_path = write_summary(rows)
    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
