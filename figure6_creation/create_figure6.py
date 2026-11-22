#!/usr/bin/env python3
"""Create the chemical-congress Figure 6 plots.

This follows pygamlab_v2_test.py. The workbook is analysed, solubility is
filtered twice by the interquartile range, and the target becomes
abs(log(solubility)). A KNN regression_PIPE then supplies the professional
true-versus-predicted plot and the three-dimensional prediction maps.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

import pygamlab.ai_core.auto_ml.exploratory_preprocessing as exploratory
import pygamlab.ai_core.auto_ml.regression_workflows as regression_workflows
from pygamlab.ai_core import Automated_Exploratory_Processor, regression_PIPE

plt.show = lambda *args, **kwargs: plt.close("all")
exploratory.plt.show = plt.show
regression_workflows.plt.show = plt.show

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "chemical_congress.xlsx"
FIGURE_DIR = ROOT / "figures"
FEATURES = ["Mw", "Melting point", "density", "HBD", "HBA"]
TARGET = "solubility"
COLUMNS = FEATURES + [TARGET]
KNN_PARAMS = {
    "model__n_neighbors": [3, 5, 7, 9, 11],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],
    "model__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "model__leaf_size": [20, 30, 40, 50],
}


def iqr_inliers(features, target):
    q1 = np.percentile(target, 25)
    q3 = np.percentile(target, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (target >= lower) & (target <= upper)
    return features[mask.flatten()], target[mask]


def prepare_congress(frame):
    features = frame[FEATURES].to_numpy()
    target = frame[TARGET].to_numpy().reshape(-1, 1)
    features, target = iqr_inliers(features, target)
    features, target = iqr_inliers(features, target)
    return features, np.abs(np.log(target))


def main():
    frame = pd.read_excel(DATA_PATH)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    analyzer = Automated_Exploratory_Processor(
        data_source=str(DATA_PATH),
        numeric_columns=COLUMNS,
    )
    os.chdir(FIGURE_DIR)
    analyzer.run_full_analysis()
    os.chdir(ROOT)

    features, target = prepare_congress(frame)
    print(f"Congress rows after two IQR filters: {len(target)}")

    workflow = regression_PIPE(features, target)
    workflow.data_clear_check()
    workflow.make_pipe(model=KNeighborsRegressor(), specific_preprocessor=MinMaxScaler())
    workflow.fitting(myparams=KNN_PARAMS)
    best_model, best_score = workflow.best()
    print(f"Best KNN score: {best_score}")
    print(f"Best parameters: {workflow.gs.best_params_}")

    os.chdir(FIGURE_DIR)
    workflow.plot_true_vs_predict(name_model="KNN", save=True, professional_plotting=True)
    workflow.plot_3d_prediction_maps(
        frame,
        "Mw",
        "Melting point",
        TARGET,
        save=True,
    )
    os.chdir(ROOT)

    print("\n" + "=" * 70)
    print("KNN MODEL - CROSS-VALIDATION METRICS")
    print("=" * 70)
    workflow.evaluate_cv_train_test_metrics()
    print(f"Best estimator: {best_model}")
    print(f"Figures written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
