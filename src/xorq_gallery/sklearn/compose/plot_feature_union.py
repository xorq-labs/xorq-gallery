"""Concatenating Multiple Feature Extraction Methods
=================================================

sklearn: Use FeatureUnion to combine PCA and SelectKBest features from the iris
dataset, then use GridSearchCV to optimize n_components, k, and SVM C parameter.

xorq: Same FeatureUnion pipeline wrapped in Pipeline.from_instance. Cross-validation
scores computed via deferred_cross_val_score with StratifiedKFold over the parameter
grid. Best parameters and score extracted via deferred execution.

Both produce identical best scores and parameters.

Dataset: Iris (sklearn)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/compose/plot_feature_union.py
"""

from __future__ import annotations

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.svm import SVC
from xorq.expr.ml.cross_validation import (
    apply_deterministic_sort,
    deferred_cross_val_score,
)
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    split_data_nop,
)
from xorq_gallery.utils import fig_to_image, save_fig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = ("sepal_length", "sepal_width", "petal_length", "petal_width")
Y_COL = "target"
ROW_IDX = "row_idx"
RANDOM_STATE = 42
N_SPLITS = 5

PARAM_GRID = {
    "features__pca__n_components": [1, 2, 3],
    "features__univ_select__k": [1, 2],
    "svm__C": [0.1, 1, 10],
}

CV_SPLITTER = StratifiedKFold(n_splits=N_SPLITS, shuffle=False)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data():
    """Load iris dataset from sklearn."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=FEATURE_COLS)
    df[Y_COL] = iris.target
    df[ROW_IDX] = range(len(df))
    return df


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------


def _build_pipeline():
    """Build FeatureUnion pipeline: PCA + SelectKBest + linear SVM."""
    combined_features = FeatureUnion(
        [("pca", PCA(n_components=2)), ("univ_select", SelectKBest(k=1))]
    )
    return SklearnPipeline(
        [("features", combined_features), ("svm", SVC(kernel="linear"))]
    )


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------


def _plot_best_score(score, title):
    """Plot best CV score as a bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["Best CV Score"], [score])
    ax.set_ylabel("Cross-Validation Score")
    ax.set_ylim(0, 1.15)
    ax.text(0, score + 0.03, f"{score:.4f}", ha="center", va="bottom", fontsize=12)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Custom make_*_result for GridSearchCV
# ---------------------------------------------------------------------------

# Use only the base pipeline as names_pipelines entry
names_pipelines = (("FeatureUnion+SVM", _build_pipeline()),)
methods = ("FeatureUnion+SVM",)


def _make_sklearn_result(
    pipeline, train_data, test_data, features, target, metrics_names_funcs
):
    """Sklearn: GridSearchCV over FeatureUnion pipeline."""
    # Sort deterministically (xorq needs this for matching fold assignments)
    con = xo.connect()
    table = con.register(train_data, "iris_sort")
    df_sorted = apply_deterministic_sort(table, random_seed=RANDOM_STATE).execute()

    X = df_sorted[list(features)].values
    y = df_sorted[target].values

    grid_search = GridSearchCV(clone(pipeline), param_grid=PARAM_GRID, cv=CV_SPLITTER)
    grid_search.fit(X, y)

    return {
        "fitted": None,
        "preds": np.array([grid_search.best_score_]),
        "metrics": {
            "best_score": grid_search.best_score_,
        },
        "other": {"best_params": grid_search.best_params_},
    }


def _make_deferred_xorq_result(
    pipeline, train_data, test_data, features, target, metrics_names_funcs, pred
):
    """Deferred xorq: manual grid search over parameter combos."""
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())

    results = []
    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        sk_pipe = _build_pipeline()
        sk_pipe.set_params(**params)
        xorq_pipe = Pipeline.from_instance(sk_pipe)
        cv_result = deferred_cross_val_score(
            xorq_pipe, train_data, features, target,
            cv=CV_SPLITTER, random_seed=RANDOM_STATE,
        )
        results.append((params, cv_result))

    return {
        "xorq_fitted": None,
        "preds": results,  # list of (params, CrossValScore)
        "metrics": {},
        "other": {},
    }


def _make_xorq_result(deferred_xorq_result):
    """Execute all deferred CV scores, find best."""
    results = deferred_xorq_result["preds"]
    best_score = -1.0
    best_params = None
    for params, cv_result in results:
        scores = cv_result.execute()
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return {
        "fitted": None,
        "preds": np.array([best_score]),
        "metrics": {"best_score": best_score},
        "other": {"best_params": best_params},
    }


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def compare_results(comparator):
    sk = comparator.sklearn_results["FeatureUnion+SVM"]
    xo_r = comparator.xorq_results["FeatureUnion+SVM"]
    sk_score = sk["metrics"]["best_score"]
    xo_score = xo_r["metrics"]["best_score"]
    sk_params = sk["other"]["best_params"]
    xo_params = xo_r["other"]["best_params"]

    print(f"  sklearn best score: {sk_score:.4f}, params: {sk_params}")
    print(f"  xorq   best score: {xo_score:.4f}, params: {xo_params}")
    np.testing.assert_allclose(sk_score, xo_score, rtol=1e-6)
    assert sk_params == xo_params
    print("Assertions passed.")


def plot_results(comparator):
    sk_score = comparator.sklearn_results["FeatureUnion+SVM"]["metrics"]["best_score"]
    xo_score = comparator.xorq_results["FeatureUnion+SVM"]["metrics"]["best_score"]

    sk_fig = _plot_best_score(sk_score, "sklearn - Grid Search Best Score")
    xo_fig = _plot_best_score(xo_score, "xorq - Grid Search Best Score")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(
        "FeatureUnion with GridSearchCV: sklearn vs xorq",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    plt.close(sk_fig)
    plt.close(xo_fig)
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=Y_COL,
    pred="pred",
    metrics_names_funcs=(),
    load_data=_load_data,
    split_data=split_data_nop,
    make_sklearn_result=_make_sklearn_result,
    make_deferred_xorq_result=_make_deferred_xorq_result,
    make_xorq_result=_make_xorq_result,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)

# Module-level deferred exprs (list of (params, cv_result) tuples)
xorq_grid_search_results = comparator.deferred_xorq_results["FeatureUnion+SVM"]["preds"]


def main():
    comparator.result_comparison
    save_fig("imgs/feature_union.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
