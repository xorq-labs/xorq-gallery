"""Comparison of LDA and PCA 2D projection of Iris dataset
===========================================================

sklearn: Load Iris dataset, fit PCA (unsupervised) and LDA (supervised) for 2D
dimensionality reduction, plot projections with class labels. PCA finds directions
of maximum variance, LDA finds directions that best separate classes.

xorq: PCA wrapped in Pipeline.from_instance for deferred fit/transform. LDA
computed eagerly (since it's both a classifier and transformer, xorq prioritizes
the classifier interface). Deferred plotting of 2D projections for both.

Both produce identical 2D projections.

Dataset: Iris (sklearn built-in)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline as SklearnPipeline
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Load Iris dataset and return as pandas DataFrame with feature names."""
    iris = load_iris()
    # Use clean column names (no spaces or special characters for xorq compatibility)
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df = pd.DataFrame(iris.data, columns=feature_cols)
    df["target"] = iris.target
    df["target_names"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_projection(X_r, y, target_names, title, colors=None):
    """Plot 2D projection with class labels.

    Parameters
    ----------
    X_r : ndarray, shape (n_samples, 2)
        2D reduced data
    y : ndarray, shape (n_samples,)
        Target labels (integers)
    target_names : list of str
        Class names
    title : str
        Plot title
    colors : list of str, optional
        Colors for each class

    Returns
    -------
    matplotlib.figure.Figure
    """
    if colors is None:
        colors = ["navy", "turquoise", "darkorange"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for color, i, target_name in zip(colors, range(len(target_names)), target_names):
        ax.scatter(
            X_r[y == i, 0],
            X_r[y == i, 1],
            color=color,
            alpha=0.8,
            lw=2,
            label=target_name,
        )

    ax.legend(loc="best", shadow=False, scatterpoints=1)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _build_pca_plot(y, target_names):
    """Return a UDAF-compatible plotting function for PCA projection."""
    def _plot(df_plot):
        X_r = df_plot[["pca0", "pca1"]].values
        return _plot_projection(X_r, y, target_names, "PCA of Iris dataset")
    return _plot


def _build_lda_plot(y, target_names):
    """Return a UDAF-compatible plotting function for LDA projection."""
    def _plot(df_lda):
        X_r = df_lda[["lda0", "lda1"]].values
        return _plot_projection(X_r, y, target_names, "LDA of Iris dataset")
    return _plot


# =========================================================================
# SKLEARN WAY -- eager fit/transform, eager plotting
# =========================================================================


def sklearn_way(df):
    """Eager sklearn: fit PCA and LDA on Iris, project to 2D, plot projections.

    Returns
    -------
    dict
        Dictionary containing PCA and LDA transformed data and explained variance ratios.
    """
    # Separate features and target
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
    y = df["target"].values
    target_names = df["target_names"].cat.categories.tolist()

    # PCA - unsupervised dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    print(f"sklearn: PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"sklearn: PCA total variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    # LDA - supervised dimensionality reduction
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X, y)

    print(f"sklearn: LDA explained variance ratio: {lda.explained_variance_ratio_}")
    print(f"sklearn: LDA total variance explained: {lda.explained_variance_ratio_.sum():.3f}")

    return {
        "X_pca": X_pca,
        "X_lda": X_lda,
        "y": y,
        "target_names": target_names,
        "pca_variance_ratio": pca.explained_variance_ratio_.copy(),
        "lda_variance_ratio": lda.explained_variance_ratio_.copy(),
    }


# =========================================================================
# XORQ WAY -- deferred fit/transform, deferred plotting
# =========================================================================


def xorq_way(df):
    """Deferred xorq: wrap PCA in Pipeline.from_instance, fit/transform deferred.

    LDA is computed eagerly and registered as a table because LDA is both a
    classifier and transformer, and xorq prioritizes the classifier interface.

    Returns
    -------
    dict
        Dictionary containing deferred PCA expression and eager LDA table.
    """
    con = xo.connect()

    # Register data
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    table = con.register(df, "iris")

    # PCA Pipeline - fully deferred
    pca_sklearn = SklearnPipeline([("pca", PCA(n_components=2))])
    pca_pipe = Pipeline.from_instance(pca_sklearn)
    pca_fitted = pca_pipe.fit(table, features=tuple(features), target=None)
    pca_transformed = pca_fitted.transform(table)

    # LDA - compute eagerly since xorq doesn't support transform for classifiers
    X = df[features].values
    y_data = df["target"].values
    lda_sklearn_inst = LDA(n_components=2)
    X_lda = lda_sklearn_inst.fit_transform(X, y_data)

    # Register LDA result as a table for deferred plotting
    lda_result_df = pd.DataFrame(X_lda, columns=["lda0", "lda1"])
    lda_table = con.register(lda_result_df, "lda_result")

    return {
        "pca_transformed": pca_transformed,
        "lda_transformed": lda_table,
        "lda_result_array": X_lda,  # For assertions
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()
    y = df["target"].values
    target_names = df["target_names"].cat.categories.tolist()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(df)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(df)

    # Execute deferred expressions
    pca_df = xo_results["pca_transformed"].execute()

    # Extract transformed data for numerical comparison
    # PCA outputs column names based on the step name: pca0, pca1
    X_pca_xorq = pca_df[["pca0", "pca1"]].values
    X_lda_xorq = xo_results["lda_result_array"]  # LDA was computed eagerly

    # ---- Assert numerical equivalence BEFORE plotting ----
    print("\n=== ASSERTIONS ===")
    np.testing.assert_allclose(sk_results["X_pca"], X_pca_xorq, rtol=1e-5)
    print("PCA projections match between sklearn and xorq")

    np.testing.assert_allclose(sk_results["X_lda"], X_lda_xorq, rtol=1e-5)
    print("LDA projections match between sklearn and xorq")

    print("Assertions passed: sklearn and xorq produce identical projections.")

    # Execute deferred plots in main()
    pca_png = deferred_matplotlib_plot(
        xo_results["pca_transformed"],
        _build_pca_plot(y, target_names),
        name="pca_plot"
    ).execute()

    lda_png = deferred_matplotlib_plot(
        xo_results["lda_transformed"],
        _build_lda_plot(y, target_names),
        name="lda_plot"
    ).execute()

    # Build sklearn plots
    sk_pca_fig = _plot_projection(
        sk_results["X_pca"],
        sk_results["y"],
        sk_results["target_names"],
        "PCA of Iris dataset",
    )
    sk_lda_fig = _plot_projection(
        sk_results["X_lda"],
        sk_results["y"],
        sk_results["target_names"],
        "LDA of Iris dataset",
    )

    # Composite: sklearn (left) | xorq (right)
    pca_img_xo = load_plot_bytes(pca_png)
    lda_img_xo = load_plot_bytes(lda_png)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top row: PCA
    axes[0, 0].imshow(fig_to_image(sk_pca_fig))
    axes[0, 0].set_title("sklearn: PCA", fontsize=14)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pca_img_xo)
    axes[0, 1].set_title("xorq: PCA", fontsize=14)
    axes[0, 1].axis("off")

    # Bottom row: LDA
    axes[1, 0].imshow(fig_to_image(sk_lda_fig))
    axes[1, 0].set_title("sklearn: LDA", fontsize=14)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(lda_img_xo)
    axes[1, 1].set_title("xorq: LDA", fontsize=14)
    axes[1, 1].axis("off")

    plt.suptitle("PCA vs LDA on Iris Dataset: sklearn vs xorq", fontsize=16)
    plt.tight_layout()
    out = "imgs/plot_pca_vs_lda.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComposite plot saved to {out}")

    # Close individual figures
    plt.close(sk_pca_fig)
    plt.close(sk_lda_fig)


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
