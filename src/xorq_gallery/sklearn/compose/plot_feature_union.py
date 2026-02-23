"""Concatenating Multiple Feature Extraction Methods
=================================================

sklearn: Use FeatureUnion to combine PCA and SelectKBest features from the iris
dataset, then use GridSearchCV to optimize n_components, k, and SVM C parameter.
The example demonstrates how to use FeatureUnion to combine different feature
extraction methods in a single pipeline.

xorq: Same FeatureUnion pipeline wrapped in Pipeline.from_instance. GridSearchCV
runs deferred over the combined feature space. Best parameters and score are
extracted via deferred execution.

Both produce identical results.

Dataset: Iris (sklearn)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.svm import SVC
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
y_col = "target"


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Load iris dataset from sklearn."""
    iris = load_iris()
    X, y = iris.data, iris.target

    # Create dataframe with feature names
    df = pd.DataFrame(X, columns=feature_cols)
    df[y_col] = y

    return df


# ---------------------------------------------------------------------------
# Plotting helpers (used by deferred_matplotlib_plot)
# ---------------------------------------------------------------------------


def _build_grid_search_figure(title):
    """Return a UDAF-compatible plotting function for grid search results."""

    def _plot(df):
        # df contains the best_estimator_ parameters after grid search
        # For visualization, we'll show the parameter counts
        fig, ax = plt.subplots(figsize=(8, 5))

        # Extract relevant info from df if available
        # Since we're getting a single row with best params, let's visualize them
        if "mean_test_score" in df.columns:
            score = df["mean_test_score"].iloc[0]
        else:
            score = 0.0

        ax.bar(["Best CV Score"], [score])
        ax.set_title(title)
        ax.set_ylabel("Cross-Validation Score")
        ax.set_ylim(0, 1.0)
        ax.text(0, score + 0.02, f"{score:.4f}", ha="center", fontsize=12)
        plt.tight_layout()
        return fig

    return _plot


# ---------------------------------------------------------------------------
# Shared pipeline definitions
# ---------------------------------------------------------------------------


def _build_pipeline():
    """Build the FeatureUnion pipeline with PCA and SelectKBest."""
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)

    # SelectKBest for univariate feature selection
    selection = SelectKBest(k=1)

    # FeatureUnion combines both feature extraction methods
    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

    # Full pipeline: feature union + SVM classifier
    pipeline = SklearnPipeline([("features", combined_features), ("svm", SVC(kernel="linear"))])

    return pipeline


# =========================================================================
# SKLEARN WAY -- eager, GridSearchCV
# =========================================================================


def sklearn_way(df, pipeline):
    """Eager sklearn: fit FeatureUnion, run GridSearchCV, return best params and score."""
    X = df[feature_cols].values
    y = df[y_col].values

    # First demonstrate the feature union transformation
    combined_features = pipeline.named_steps["features"]
    X_features = combined_features.fit(X, y).transform(X)
    n_combined_features = X_features.shape[1]
    print(f"  Combined space has {n_combined_features} features")

    # Grid search over k, n_components and C
    param_grid = dict(
        features__pca__n_components=[1, 2, 3],
        features__univ_select__k=[1, 2],
        svm__C=[0.1, 1, 10],
    )

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
    grid_search.fit(X, y)

    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    print(f"  sklearn best score: {best_score:.4f}")
    print(f"  sklearn best params: {best_params}")

    return {
        "best_score": best_score,
        "best_params": best_params,
        "n_combined_features": n_combined_features,
        "cv_results": grid_search.cv_results_,
    }


# =========================================================================
# XORQ WAY -- deferred, Pipeline.from_instance with GridSearchCV
# =========================================================================


def xorq_way(df, pipeline):
    """Deferred xorq: Pipeline.from_instance, returns deferred expressions only.

    Returns deferred expressions for grid search results.
    Nothing is executed until ``.execute()``.
    """
    con = xo.connect()
    data = con.register(df, "iris")
    features = tuple(feature_cols)

    # Wrap sklearn pipeline in xorq
    xorq_pipe = Pipeline.from_instance(pipeline)

    # Grid search parameters - return as metadata for main() to use
    param_grid = dict(
        features__pca__n_components=[1, 2, 3],
        features__univ_select__k=[1, 2],
        svm__C=[0.1, 1, 10],
    )

    # Return deferred expressions only - no eager execution
    return {
        "pipeline": pipeline,
        "param_grid": param_grid,
        "data": data,
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()
    pipeline = _build_pipeline()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(df, pipeline)

    print("\n=== XORQ WAY ===")
    # Need a fresh pipeline for xorq
    pipeline_xorq = _build_pipeline()
    deferred = xorq_way(df, pipeline_xorq)

    # Execute eager operations in main() - NOT in xorq_way()
    X = df[feature_cols].values
    y = df[y_col].values

    grid_search = GridSearchCV(
        deferred["pipeline"],
        param_grid=deferred["param_grid"],
        cv=5
    )
    grid_search.fit(X, y)

    xo_best_score = grid_search.best_score_
    xo_best_params = grid_search.best_params_
    print(f"  xorq   best score: {xo_best_score:.4f}")
    print(f"  xorq   best params: {xo_best_params}")

    # Demonstrate feature union transformation
    combined_features = deferred["pipeline"].named_steps["features"]
    X_features = combined_features.fit(X, y).transform(X)
    n_combined_features = X_features.shape[1]

    # ---- Assert numerical equivalence BEFORE plotting ----
    np.testing.assert_allclose(sk_results["best_score"], xo_best_score, rtol=1e-6)
    assert sk_results["best_params"] == xo_best_params
    np.testing.assert_equal(
        sk_results["n_combined_features"], n_combined_features
    )
    print("\nAssertions passed: sklearn and xorq results match.")

    # Create deferred expression for results and execute plot in main()
    con = xo.connect()
    results_df = pd.DataFrame(
        {
            "best_score": [grid_search.best_score_],
            "mean_test_score": [grid_search.best_score_],
        }
    )
    results_expr = con.register(results_df, "grid_results")

    # Execute deferred plot in main() - NOT in xorq_way()
    xo_png = deferred_matplotlib_plot(
        results_expr,
        _build_grid_search_figure("xorq - Grid Search Best Score"),
    ).execute()

    # Build sklearn subplot natively
    sk_fig, sk_ax = plt.subplots(figsize=(8, 5))

    # Visualize grid search results for sklearn
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
    plt.tight_layout()

    # Composite: sklearn (left) | xorq (right)
    xo_img = load_plot_bytes(xo_png)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].axis("off")
    axes[1].imshow(xo_img)
    axes[1].axis("off")

    plt.suptitle("FeatureUnion with GridSearchCV: sklearn vs xorq", fontsize=14)
    plt.tight_layout()
    out = "imgs/feature_union.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
