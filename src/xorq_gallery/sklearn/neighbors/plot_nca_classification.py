"""Comparing Nearest Neighbors with and without Neighborhood Components Analysis
=================================================================================

sklearn: Load iris dataset (2 features), train KNN and NCA+KNN pipelines with
StandardScaler, evaluate accuracy on stratified test split, plot decision
boundaries.

xorq: Same pipelines wrapped in Pipeline.from_instance, fit/predict deferred,
accuracy via deferred_sklearn_metric. Decision boundaries use xorq fitted
pipeline for meshgrid prediction via _XorqClfWrapper.

Both produce identical accuracy scores.

Dataset: iris (sklearn)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
)
from xorq_gallery.sklearn.sklearn_lib import (
    make_deferred_xorq_result as _make_deferred_xorq_result,
)
from xorq_gallery.sklearn.sklearn_lib import (
    make_sklearn_result as _make_sklearn_result,
)
from xorq_gallery.utils import save_fig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_NEIGHBORS = 1
RANDOM_STATE = 42
TEST_SIZE = 0.7
H = 0.05  # meshgrid step size

FEATURE_COLS = ("x0", "x1")
TARGET_COL = "y"
PRED_COL = "pred"

CMAP_LIGHT = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
CMAP_BOLD = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Load iris dataset, select sepal length and petal length (2 features)."""
    dataset = datasets.load_iris()
    X = dataset.data[:, [0, 2]]
    return pd.DataFrame(X, columns=list(FEATURE_COLS)).assign(
        **{TARGET_COL: dataset.target}
    )


def split_data(df):
    """Stratified train/test split."""
    return train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COL]
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_decision_boundary(ax, X, y, clf, title, score):
    """Plot decision boundary and data points for a classifier."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, H), np.arange(y_min, y_max, H))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=CMAP_LIGHT, alpha=0.8, shading="auto")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CMAP_BOLD, edgecolors="k", s=20)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)
    ax.text(
        0.9,
        0.1,
        f"{score:.2f}",
        size=15,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def compare_results(comparator):
    assert sorted(sklearn_results := comparator.sklearn_results) == sorted(
        xorq_results := comparator.xorq_results
    )
    print("\n=== Comparing Results ===")
    for name, sklearn_result in sklearn_results.items():
        xorq_result = xorq_results[name]
        sk_acc = sklearn_result["metrics"]["accuracy"]
        xo_acc = xorq_result["metrics"]["accuracy"]
        print(f"  {name:10s} accuracy - sklearn: {sk_acc:.4f}, xorq: {xo_acc:.4f}")


def plot_results(comparator):
    X = comparator.df[list(FEATURE_COLS)].values
    y = comparator.df[TARGET_COL].values

    fig, axes = plt.subplots(2, len(methods), figsize=(12, 10))

    for col, name in enumerate(methods):
        sk_score = comparator.sklearn_results[name]["metrics"]["accuracy"]
        xo_score = comparator.xorq_results[name]["metrics"]["accuracy"]

        # "other" stores the full fitted pipeline (StandardScaler + [NCA +] KNN)
        sk_clf = comparator.sklearn_results[name]["other"]["full_pipeline"]
        xo_clf = comparator.xorq_results[name]["other"]["full_pipeline"]

        # Top row: sklearn
        _plot_decision_boundary(axes[0, col], X, y, sk_clf, name, sk_score)
        if col == 0:
            axes[0, col].set_ylabel("sklearn", fontsize=12, fontweight="bold")

        # Bottom row: xorq (fitted model is same sklearn pipeline)
        _plot_decision_boundary(axes[1, col], X, y, xo_clf, name, xo_score)
        if col == 0:
            axes[1, col].set_ylabel("xorq", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Comparing Nearest Neighbors with/without NCA: sklearn vs xorq", fontsize=16
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# make_other override: store full fitted pipeline for decision boundary plotting
# ---------------------------------------------------------------------------


def _make_sklearn_other(fitted):
    return {"full_pipeline": fitted}


def _make_xorq_other(xorq_fitted):
    """Reconstruct a full sklearn Pipeline from xorq's fitted steps.

    Stored as a lambda so make_xorq_result can call v() to materialize.
    """
    steps = [(step.step.name, step.model) for step in xorq_fitted.fitted_steps]
    full_pipeline = SklearnPipeline(steps)
    return {"full_pipeline": lambda: full_pipeline}


make_sklearn_result = _make_sklearn_result(make_other=_make_sklearn_other)
make_deferred_xorq_result = _make_deferred_xorq_result(make_other=_make_xorq_other)


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = (KNN_NAME, NCA_KNN_NAME) = ("KNN", "NCA+KNN")
names_pipelines = (
    (
        KNN_NAME,
        SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=N_NEIGHBORS)),
            ]
        ),
    ),
    (
        NCA_KNN_NAME,
        SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("nca", NeighborhoodComponentsAnalysis(random_state=RANDOM_STATE)),
                ("knn", KNeighborsClassifier(n_neighbors=N_NEIGHBORS)),
            ]
        ),
    ),
)
metrics_names_funcs = (("accuracy", accuracy_score),)

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=TARGET_COL,
    pred=PRED_COL,
    metrics_names_funcs=metrics_names_funcs,
    load_data=load_data,
    split_data=split_data,
    make_sklearn_result=make_sklearn_result,
    make_deferred_xorq_result=make_deferred_xorq_result,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)
# expose the exprs to invoke `xorq build plot_nca_classification.py --expr $expr_name`
(xorq_knn_preds, xorq_nca_knn_preds) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    comparator.result_comparison
    save_fig("imgs/plot_nca_classification.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
