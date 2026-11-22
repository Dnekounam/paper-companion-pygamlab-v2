#!/usr/bin/env python3
"""Create the copper-nanocomposite CTE models with pygamlab AutoML, then package them.

This is the CTE section of ai_training_regression.py. The workbook supplies
graphene weight fraction ``w`` and temperature ``t``; the target is the thermal
expansion coefficient ``alpha``. pygamlab AutoML trains each estimator, and
``GamAI_io`` writes the trained object to a ``.gam_ai`` file.

The train/test models use ``BASIC_regression``. The searched models use
``regression_PIPE`` with the original parameter grids. The packaged names are
the names shipped in pygamlab.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import pygamlab.ai_core.auto_ml.regression_workflows as regression_workflows
from pygamlab.ai_core import BASIC_regression, GamAI_io, regression_PIPE

plt.show = lambda *args, **kwargs: plt.close("all")
regression_workflows.plt.show = plt.show

DATA_PATH = Path(__file__).resolve().parent / "cte_cu_nanocomposite.xlsx"
MODEL_DIR = Path(__file__).resolve().parent / "models"
FEATURES = ["w", "t"]
TARGET = "alpha"
DESCRIPTION = (
    "The presence of vacancy defects in graphene negatively affects the "
    "structural behaviors of the composite beams to a certain degree. So, "
    "increasing the temperature & defects decrease the mechanical properties, "
    "Increasing the percentage of Graphene increase the mechanical properties."
)
AUTHOR_NAME = "Shaoyu Zhao, Yingyan Zhang, Yihe Zhang et al."
TRAINER_NAME = "Ali Pilehvar Meibody"
DOI = "https://doi.org/10.1007/s00366-022-01710-w"

KNN_PARAMS = {
    "model__n_neighbors": [3, 5, 7, 9, 11],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],
    "model__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "model__leaf_size": [20, 30, 40, 50],
}
DT_PARAMS = {
    "model__criterion": ["squared_error", "absolute_error", "friedman_mse"],
    "model__splitter": ["best", "random"],
    "model__max_depth": [None, 5, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": [None, "sqrt", "log2"],
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
SVR_PARAMS = {
    "model__kernel": ["linear", "poly", "rbf", "sigmoid"],
    "model__C": [0.1, 1, 10, 100],
    "model__epsilon": [0.01, 0.1, 0.2, 0.5],
    "model__gamma": ["scale", "auto"],
}
MLP_PARAMS = {
    "model__hidden_layer_sizes": [(50,), (100,), (100, 50), (100, 100)],
    "model__activation": ["relu", "tanh", "logistic"],
    "model__solver": ["adam", "lbfgs"],
    "model__alpha": [0.0001, 0.001, 0.01],
    "model__learning_rate": ["constant", "adaptive"],
    "model__max_iter": [500, 1000],
}


def load_cte():
    frame = pd.read_excel(DATA_PATH)
    features = frame[FEATURES].astype(float).to_numpy()
    target = frame[TARGET].astype(float).to_numpy().reshape(-1, 1)
    return features, target


def package(model_name, model_type, accuracy, trained_model, search=None, best_params=None):
    record = GamAI_io(
        model_name=model_name,
        model_type=model_type,
        description=DESCRIPTION,
        author_name=AUTHOR_NAME,
        author_email="None",
        trainer_name=TRAINER_NAME,
        best_accuracy=accuracy,
        doi=DOI,
        hyperparam_range=search,
        best_params=best_params,
        ml_model=trained_model,
    )
    record.save(save_dir=str(MODEL_DIR))
    return MODEL_DIR / f"{model_name}.gam_ai"


def create_train_test_model(features, target, model_name, estimator, plot_name):
    """AutoML train/test path: split, scale, fit, score, then package."""
    workflow = BASIC_regression(features, target)
    workflow.split(test_size=0.2)
    workflow.scaling(def_scaler="minmax")
    workflow.fitting(model=estimator)
    score = workflow.get_score()
    workflow.plot_true_vs_predict(name_model=plot_name)
    path = package(
        model_name,
        "train/test",
        score,
        workflow.model,
    )
    print(f"created {path.name}: R2 {score['R2']:.6f}")
    return path


def create_pipe_model(features, target, model_name, estimator, search):
    """AutoML pipeline path: scale, grid search, then package the search object."""
    workflow = regression_PIPE(features, target)
    workflow.data_clear_check()
    workflow.make_pipe(model=estimator, specific_preprocessor=MinMaxScaler())
    workflow.fitting(myparams=search)
    best_model, best_score = workflow.best()
    workflow.plot_true_vs_predict(name_model=type(estimator).__name__)
    path = package(
        model_name,
        "pipe",
        best_score,
        workflow.gs,
        search=search,
        best_params=workflow.gs.best_params_,
    )
    print(f"created {path.name}: CV score {best_score:.6f}; best {best_model}")
    return path


def main():
    features, target = load_cte()
    print(f"CTE data: {features.shape[0]} rows, features {FEATURES}, target {TARGET}")

    train_test_models = (
        ("cu-nanocomposites-cte-lr", LinearRegression(), "Linear Regression"),
        ("cu-nanocomposites-cte-knn", KNeighborsRegressor(), "KNN"),
        ("cu-nanocomposites-cte-dt", DecisionTreeRegressor(), "Decision Tree"),
        ("cu-nanocomposites-cte-rf", RandomForestRegressor(), "Random forest"),
        ("cu-nanocomposites-cte-svr", SVR(), "SVR"),
        ("cu-nanocomposites-cte-mlp", MLPRegressor(hidden_layer_sizes=(100, 10)), "MLP"),
    )
    for model_name, estimator, plot_name in train_test_models:
        create_train_test_model(features, target, model_name, estimator, plot_name)

    pipe_models = (
        ("cu-nanocomposites-cte-pipe-lr", LinearRegression(), {}),
        ("cu-nanocomposites-cte-pipe-knn", KNeighborsRegressor(), KNN_PARAMS),
        ("cu-nanocomposites-cte-pipe-dt", DecisionTreeRegressor(), DT_PARAMS),
        ("cu-nanocomposites-cte-pipe-rf", RandomForestRegressor(), RF_PARAMS),
        ("cu-nanocomposites-cte-pipe-svr", SVR(), SVR_PARAMS),
        ("cu-nanocomposites-cte-pipe-mlp", MLPRegressor(), MLP_PARAMS),
    )
    for model_name, estimator, search in pipe_models:
        create_pipe_model(features, target, model_name, estimator, search)

    saved = sorted(MODEL_DIR.glob("*.gam_ai"))
    print(f"\n{len(saved)} .gam_ai packages in {MODEL_DIR}")


if __name__ == "__main__":
    main()
