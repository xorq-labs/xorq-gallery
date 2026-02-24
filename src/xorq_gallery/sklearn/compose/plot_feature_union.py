"""Concatenating Multiple Feature Extraction Methods
=================================================

sklearn: Use FeatureUnion to combine PCA and SelectKBest features from the iris
dataset, then use GridSearchCV to optimize n_components, k, and SVM C parameter.

xorq: Same FeatureUnion pipeline wrapped in Pipeline.from_instance. Cross-validation
scores computed via deferred_cross_val_score with StratifiedKFold over the parameter
grid. Best parameters and score extracted via deferred execution.

Both produce identical best scores and parameters.

Dataset: Iris (sklearn)
"""

from __future__ import annotations

import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.svm import SVC
from toolz import curry
from xorq.expr.ml.cross_validation import (
    apply_deterministic_sort,
    deferred_cross_val_score,
)
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


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

# Shared splitter — same object for sklearn GridSearchCV and xorq deferred CV
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


@curry
def _grid_search_figure(df, title):
    """Plot best CV score as a bar chart."""
    score = df["best_score"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["Best CV Score"], [score])
    ax.set_title(title)
    ax.set_ylabel("Cross-Validation Score")
    ax.set_ylim(0, 1.0)
    ax.text(0, score + 0.02, f"{score:.4f}", ha="center", fontsize=12)
    fig.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager GridSearchCV
# =========================================================================


def sklearn_way(df_sorted, pipeline):
    """Eager sklearn: GridSearchCV over FeatureUnion pipeline.

    df_sorted must already be sorted by apply_deterministic_sort so that
    StratifiedKFold fold assignments match the xorq side.
    """
    X = df_sorted[list(FEATURE_COLS)].values
    y = df_sorted[Y_COL].values

    # Demonstrate feature union transformation
    combined_features = pipeline.named_steps["features"]
    X_features = combined_features.fit(X, y).transform(X)
    n_combined_features = X_features.shape[1]
    print(f"  Combined space has {n_combined_features} features")

    grid_search = GridSearchCV(pipeline, param_grid=PARAM_GRID, cv=CV_SPLITTER)
    grid_search.fit(X, y)

    print(f"  sklearn best score: {grid_search.best_score_:.4f}")
    print(f"  sklearn best params: {grid_search.best_params_}")

    return {
        "best_score": grid_search.best_score_,
        "best_params": grid_search.best_params_,
        "n_combined_features": n_combined_features,
    }


# =========================================================================
# XORQ WAY -- deferred cross-validation over parameter grid
# =========================================================================


def xorq_way(data):
    """Deferred xorq: iterate parameter grid, deferred_cross_val_score per combo.

    Note: This is a manual grid search -- we iterate over all parameter
    combinations and run deferred_cross_val_score for each.  A deferred
    GridSearchCV equivalent does not yet exist in xorq, so this hand-rolled
    loop is the current recommended approach.

    Returns list of (params_dict, CrossValScore) tuples.
    Nothing is executed until .execute().
    """
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())

    results = []
    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        sk_pipe = _build_pipeline()
        sk_pipe.set_params(**params)
        xorq_pipe = Pipeline.from_instance(sk_pipe)
        cv_result = deferred_cross_val_score(
            xorq_pipe,
            data,
            FEATURE_COLS,
            Y_COL,
            cv=CV_SPLITTER,
            random_seed=RANDOM_STATE,
        )
        results.append((params, cv_result))

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()

    con = xo.connect()
    table = con.register(df, "iris")

    # Sort by deterministic hash so sklearn's StratifiedKFold sees the same
    # row order as xorq's deferred CV
    df_sorted = apply_deterministic_sort(table, random_seed=RANDOM_STATE).execute()

    print("=== SKLEARN WAY ===")
    sk_pipeline = _build_pipeline()
    sk_results = sklearn_way(df_sorted, sk_pipeline)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(table)

    # Execute all deferred CV scores, find best
    best_score = -1.0
    best_params = None
    for params, cv_result in xo_results:
        scores = cv_result.execute()
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    print(f"  xorq   best score: {best_score:.4f}")
    print(f"  xorq   best params: {best_params}")

    # Assert
    print("\n=== ASSERTIONS ===")
    np.testing.assert_allclose(sk_results["best_score"], best_score, rtol=1e-6)
    assert sk_results["best_params"] == best_params
    print("Assertions passed: sklearn and xorq results match.")

    # Plot
    print("\n=== PLOTTING ===")
    results_df = pd.DataFrame({"best_score": [best_score]})
    results_expr = con.register(results_df, "grid_results")

    xo_png = deferred_matplotlib_plot(
        results_expr,
        _grid_search_figure(title="xorq - Grid Search Best Score"),
    ).execute()

    sk_fig, sk_ax = plt.subplots(figsize=(8, 5))
    sk_ax.bar(["Best CV Score"], [sk_results["best_score"]])
    sk_ax.set_title("sklearn - Grid Search Best Score")
    sk_ax.set_ylabel("Cross-Validation Score")
    sk_ax.set_ylim(0, 1.0)
    sk_ax.text(
        0,
        sk_results["best_score"] + 0.02,
        f"{sk_results['best_score']:.4f}",
        ha="center",
        fontsize=12,
    )
    sk_fig.tight_layout()

    xo_img = load_plot_bytes(xo_png)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(xo_img)
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(
        "FeatureUnion with GridSearchCV: sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    out = "imgs/feature_union.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Composite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
