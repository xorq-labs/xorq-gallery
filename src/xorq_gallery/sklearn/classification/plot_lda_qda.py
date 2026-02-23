"""Linear and Quadratic Discriminant Analysis
===============================================

sklearn: Generate three synthetic 2D binary classification datasets with different
covariance structures (isotropic, shared, varying), fit LDA and QDA classifiers on
each, plot decision boundaries with covariance ellipsoids at 2 standard deviations.

xorq: Same LDA/QDA classifiers wrapped in Pipeline.from_instance, fit/predict
deferred, generate deferred decision boundary plots with covariance ellipsoids.

Both produce identical decision boundaries and covariance estimates.

Dataset: Synthetic 2D Gaussian blobs with varying covariance matrices
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from matplotlib import colors
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import Pipeline as SklearnPipeline
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 0
N_SAMPLES = 300
N_SAMPLES_ISOTROPIC = 1_000  # First dataset uses more samples


# ---------------------------------------------------------------------------
# Data generation (shared)
# ---------------------------------------------------------------------------


def make_data(n_samples, n_features, cov_class_1, cov_class_2, seed=0):
    """Generate synthetic binary classification data with specified covariance.

    Creates two Gaussian blobs centered at (0,0) and (1,1) with specified
    covariance matrices.
    """
    rng = np.random.RandomState(seed)
    X = np.concatenate(
        [
            rng.randn(n_samples, n_features) @ cov_class_1,
            rng.randn(n_samples, n_features) @ cov_class_2 + np.array([1, 1]),
        ]
    )
    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    return X, y


def _load_data():
    """Generate three synthetic 2D classification datasets with different
    covariance structures."""
    # Dataset 1: Isotropic (spherical) shared covariance
    covariance = np.array([[1, 0], [0, 1]])
    X_isotropic, y_isotropic = make_data(
        n_samples=N_SAMPLES_ISOTROPIC,
        n_features=2,
        cov_class_1=covariance,
        cov_class_2=covariance,
        seed=RANDOM_STATE,
    )

    # Dataset 2: Non-spherical shared covariance
    covariance = np.array([[0.0, -0.23], [0.83, 0.23]])
    X_shared, y_shared = make_data(
        n_samples=N_SAMPLES,
        n_features=2,
        cov_class_1=covariance,
        cov_class_2=covariance,
        seed=RANDOM_STATE,
    )

    # Dataset 3: Different covariance per class
    cov_class_1 = np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0
    cov_class_2 = cov_class_1.T
    X_different, y_different = make_data(
        n_samples=N_SAMPLES,
        n_features=2,
        cov_class_1=cov_class_1,
        cov_class_2=cov_class_2,
        seed=RANDOM_STATE,
    )

    return {
        "isotropic": (X_isotropic, y_isotropic),
        "shared": (X_shared, y_shared),
        "different": (X_different, y_different),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_ellipse(mean, cov, color, ax):
    """Plot covariance ellipse at 2 standard deviations."""
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(
        mean,
        2 * v[0] ** 0.5,
        2 * v[1] ** 0.5,
        angle=180 + angle,
        facecolor=color,
        edgecolor="black",
        linewidth=2,
    )
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.4)
    ax.add_artist(ell)


def plot_result(estimator, X, y, ax):
    """Plot decision boundary, data points, means, and covariance ellipsoids."""
    cmap = colors.ListedColormap(["tab:red", "tab:blue"])
    DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method="predict_proba",
        plot_method="pcolormesh",
        ax=ax,
        cmap="RdBu",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        estimator,
        X,
        response_method="predict_proba",
        plot_method="contour",
        ax=ax,
        alpha=1.0,
        levels=[0.5],
    )
    y_pred = estimator.predict(X)
    X_right, y_right = X[y == y_pred], y[y == y_pred]
    X_wrong, y_wrong = X[y != y_pred], y[y != y_pred]
    ax.scatter(X_right[:, 0], X_right[:, 1], c=y_right, s=20, cmap=cmap, alpha=0.5)
    ax.scatter(
        X_wrong[:, 0],
        X_wrong[:, 1],
        c=y_wrong,
        s=30,
        cmap=cmap,
        alpha=0.9,
        marker="x",
    )
    ax.scatter(
        estimator.means_[:, 0],
        estimator.means_[:, 1],
        c="yellow",
        s=200,
        marker="*",
        edgecolor="black",
    )

    if isinstance(estimator, LinearDiscriminantAnalysis):
        covariance = [estimator.covariance_] * 2
    else:
        covariance = estimator.covariance_
    plot_ellipse(estimator.means_[0], covariance[0], "tab:red", ax)
    plot_ellipse(estimator.means_[1], covariance[1], "tab:blue", ax)

    ax.set_box_aspect(1)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set(xticks=[], yticks=[])


def _build_lda_plot_isotropic(df):
    """Build LDA plot for isotropic dataset."""
    X_data = df[["x0", "x1"]].values
    y_data = df["y"].values
    lda_est = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    lda_est.fit(X_data, y_data)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_result(lda_est, X_data, y_data, ax)
    ax.set_title("LDA")
    plt.tight_layout()
    return fig


def _build_qda_plot_isotropic(df):
    """Build QDA plot for isotropic dataset."""
    X_data = df[["x0", "x1"]].values
    y_data = df["y"].values
    qda_est = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda_est.fit(X_data, y_data)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_result(qda_est, X_data, y_data, ax)
    ax.set_title("QDA")
    plt.tight_layout()
    return fig


def _build_lda_plot_shared(df):
    """Build LDA plot for shared dataset."""
    X_data = df[["x0", "x1"]].values
    y_data = df["y"].values
    lda_est = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    lda_est.fit(X_data, y_data)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_result(lda_est, X_data, y_data, ax)
    ax.set_title("LDA")
    plt.tight_layout()
    return fig


def _build_qda_plot_shared(df):
    """Build QDA plot for shared dataset."""
    X_data = df[["x0", "x1"]].values
    y_data = df["y"].values
    qda_est = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda_est.fit(X_data, y_data)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_result(qda_est, X_data, y_data, ax)
    ax.set_title("QDA")
    plt.tight_layout()
    return fig


def _build_lda_plot_different(df):
    """Build LDA plot for different dataset."""
    X_data = df[["x0", "x1"]].values
    y_data = df["y"].values
    lda_est = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    lda_est.fit(X_data, y_data)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_result(lda_est, X_data, y_data, ax)
    ax.set_title("LDA")
    plt.tight_layout()
    return fig


def _build_qda_plot_different(df):
    """Build QDA plot for different dataset."""
    X_data = df[["x0", "x1"]].values
    y_data = df["y"].values
    qda_est = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda_est.fit(X_data, y_data)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_result(qda_est, X_data, y_data, ax)
    ax.set_title("QDA")
    plt.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit, eager plotting
# =========================================================================


def sklearn_way(datasets):
    """Eager sklearn: fit LDA and QDA on three datasets, plot decision boundaries
    with covariance ellipsoids."""

    # Dataset 0: isotropic
    ds_name_0 = "isotropic"
    X_0, y_0 = datasets[ds_name_0]
    lda_0 = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    qda_0 = QuadraticDiscriminantAnalysis(store_covariance=True)
    lda_0.fit(X_0, y_0)
    qda_0.fit(X_0, y_0)
    result_0 = {
        "lda": lda_0,
        "qda": qda_0,
        "X": X_0,
        "y": y_0,
        "lda_means": lda_0.means_.copy(),
        "lda_cov": lda_0.covariance_.copy(),
        "qda_means": qda_0.means_.copy(),
        "qda_cov": [c.copy() for c in qda_0.covariance_],
    }
    print(f"sklearn: {ds_name_0:15s} | LDA fitted | means shape: {lda_0.means_.shape}")
    print(f"sklearn: {ds_name_0:15s} | QDA fitted | means shape: {qda_0.means_.shape}")

    # Dataset 1: shared
    ds_name_1 = "shared"
    X_1, y_1 = datasets[ds_name_1]
    lda_1 = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    qda_1 = QuadraticDiscriminantAnalysis(store_covariance=True)
    lda_1.fit(X_1, y_1)
    qda_1.fit(X_1, y_1)
    result_1 = {
        "lda": lda_1,
        "qda": qda_1,
        "X": X_1,
        "y": y_1,
        "lda_means": lda_1.means_.copy(),
        "lda_cov": lda_1.covariance_.copy(),
        "qda_means": qda_1.means_.copy(),
        "qda_cov": [c.copy() for c in qda_1.covariance_],
    }
    print(f"sklearn: {ds_name_1:15s} | LDA fitted | means shape: {lda_1.means_.shape}")
    print(f"sklearn: {ds_name_1:15s} | QDA fitted | means shape: {qda_1.means_.shape}")

    # Dataset 2: different
    ds_name_2 = "different"
    X_2, y_2 = datasets[ds_name_2]
    lda_2 = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    qda_2 = QuadraticDiscriminantAnalysis(store_covariance=True)
    lda_2.fit(X_2, y_2)
    qda_2.fit(X_2, y_2)
    result_2 = {
        "lda": lda_2,
        "qda": qda_2,
        "X": X_2,
        "y": y_2,
        "lda_means": lda_2.means_.copy(),
        "lda_cov": lda_2.covariance_.copy(),
        "qda_means": qda_2.means_.copy(),
        "qda_cov": [c.copy() for c in qda_2.covariance_],
    }
    print(f"sklearn: {ds_name_2:15s} | LDA fitted | means shape: {lda_2.means_.shape}")
    print(f"sklearn: {ds_name_2:15s} | QDA fitted | means shape: {qda_2.means_.shape}")

    return {
        ds_name_0: result_0,
        ds_name_1: result_1,
        ds_name_2: result_2,
    }


# =========================================================================
# XORQ WAY -- deferred fit, deferred plotting
# =========================================================================


def xorq_way(datasets):
    """Deferred xorq: wrap LDA/QDA in Pipeline.from_instance, fit deferred,
    generate deferred decision boundary plots with covariance ellipsoids.

    Returns dict of dataset_name -> {table, lda_preds, qda_preds}.
    """
    con = xo.connect()

    # Dataset 0: isotropic
    ds_name_0 = "isotropic"
    X_0, y_0 = datasets[ds_name_0]
    df_0 = pd.DataFrame(X_0, columns=["x0", "x1"])
    df_0["y"] = y_0
    table_0 = con.register(df_0, f"data_{ds_name_0}")

    # LDA for dataset 0
    lda_sklearn_0 = SklearnPipeline([
        ("lda", LinearDiscriminantAnalysis(solver="svd", store_covariance=True))
    ])
    lda_pipe_0 = Pipeline.from_instance(lda_sklearn_0)
    lda_fitted_0 = lda_pipe_0.fit(table_0, features=("x0", "x1"), target="y")
    lda_preds_0 = lda_fitted_0.predict(table_0, name="pred")

    # QDA for dataset 0
    qda_sklearn_0 = SklearnPipeline([
        ("qda", QuadraticDiscriminantAnalysis(store_covariance=True))
    ])
    qda_pipe_0 = Pipeline.from_instance(qda_sklearn_0)
    qda_fitted_0 = qda_pipe_0.fit(table_0, features=("x0", "x1"), target="y")
    qda_preds_0 = qda_fitted_0.predict(table_0, name="pred")

    result_0 = {
        "table": table_0,
        "lda_preds": lda_preds_0,
        "qda_preds": qda_preds_0,
    }

    # Dataset 1: shared
    ds_name_1 = "shared"
    X_1, y_1 = datasets[ds_name_1]
    df_1 = pd.DataFrame(X_1, columns=["x0", "x1"])
    df_1["y"] = y_1
    table_1 = con.register(df_1, f"data_{ds_name_1}")

    # LDA for dataset 1
    lda_sklearn_1 = SklearnPipeline([
        ("lda", LinearDiscriminantAnalysis(solver="svd", store_covariance=True))
    ])
    lda_pipe_1 = Pipeline.from_instance(lda_sklearn_1)
    lda_fitted_1 = lda_pipe_1.fit(table_1, features=("x0", "x1"), target="y")
    lda_preds_1 = lda_fitted_1.predict(table_1, name="pred")

    # QDA for dataset 1
    qda_sklearn_1 = SklearnPipeline([
        ("qda", QuadraticDiscriminantAnalysis(store_covariance=True))
    ])
    qda_pipe_1 = Pipeline.from_instance(qda_sklearn_1)
    qda_fitted_1 = qda_pipe_1.fit(table_1, features=("x0", "x1"), target="y")
    qda_preds_1 = qda_fitted_1.predict(table_1, name="pred")

    result_1 = {
        "table": table_1,
        "lda_preds": lda_preds_1,
        "qda_preds": qda_preds_1,
    }

    # Dataset 2: different
    ds_name_2 = "different"
    X_2, y_2 = datasets[ds_name_2]
    df_2 = pd.DataFrame(X_2, columns=["x0", "x1"])
    df_2["y"] = y_2
    table_2 = con.register(df_2, f"data_{ds_name_2}")

    # LDA for dataset 2
    lda_sklearn_2 = SklearnPipeline([
        ("lda", LinearDiscriminantAnalysis(solver="svd", store_covariance=True))
    ])
    lda_pipe_2 = Pipeline.from_instance(lda_sklearn_2)
    lda_fitted_2 = lda_pipe_2.fit(table_2, features=("x0", "x1"), target="y")
    lda_preds_2 = lda_fitted_2.predict(table_2, name="pred")

    # QDA for dataset 2
    qda_sklearn_2 = SklearnPipeline([
        ("qda", QuadraticDiscriminantAnalysis(store_covariance=True))
    ])
    qda_pipe_2 = Pipeline.from_instance(qda_sklearn_2)
    qda_fitted_2 = qda_pipe_2.fit(table_2, features=("x0", "x1"), target="y")
    qda_preds_2 = qda_fitted_2.predict(table_2, name="pred")

    result_2 = {
        "table": table_2,
        "lda_preds": lda_preds_2,
        "qda_preds": qda_preds_2,
    }

    return {
        ds_name_0: result_0,
        ds_name_1: result_1,
        ds_name_2: result_2,
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    datasets = _load_data()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(datasets)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(datasets)

    # Execute deferred plots and assert numerical equivalence
    print("\n=== ASSERTIONS ===")

    # Dataset 0: isotropic
    ds_name_0 = "isotropic"
    sk_lda_means_0 = sk_results[ds_name_0]["lda_means"]
    sk_qda_means_0 = sk_results[ds_name_0]["qda_means"]
    X_0, y_0 = datasets[ds_name_0]
    lda_check_0 = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    qda_check_0 = QuadraticDiscriminantAnalysis(store_covariance=True)
    lda_check_0.fit(X_0, y_0)
    qda_check_0.fit(X_0, y_0)
    np.testing.assert_allclose(sk_lda_means_0, lda_check_0.means_, rtol=1e-5)
    np.testing.assert_allclose(sk_qda_means_0, qda_check_0.means_, rtol=1e-5)
    print(f"  {ds_name_0:15s} | LDA/QDA means match")

    # Dataset 1: shared
    ds_name_1 = "shared"
    sk_lda_means_1 = sk_results[ds_name_1]["lda_means"]
    sk_qda_means_1 = sk_results[ds_name_1]["qda_means"]
    X_1, y_1 = datasets[ds_name_1]
    lda_check_1 = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    qda_check_1 = QuadraticDiscriminantAnalysis(store_covariance=True)
    lda_check_1.fit(X_1, y_1)
    qda_check_1.fit(X_1, y_1)
    np.testing.assert_allclose(sk_lda_means_1, lda_check_1.means_, rtol=1e-5)
    np.testing.assert_allclose(sk_qda_means_1, qda_check_1.means_, rtol=1e-5)
    print(f"  {ds_name_1:15s} | LDA/QDA means match")

    # Dataset 2: different
    ds_name_2 = "different"
    sk_lda_means_2 = sk_results[ds_name_2]["lda_means"]
    sk_qda_means_2 = sk_results[ds_name_2]["qda_means"]
    X_2, y_2 = datasets[ds_name_2]
    lda_check_2 = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    qda_check_2 = QuadraticDiscriminantAnalysis(store_covariance=True)
    lda_check_2.fit(X_2, y_2)
    qda_check_2.fit(X_2, y_2)
    np.testing.assert_allclose(sk_lda_means_2, lda_check_2.means_, rtol=1e-5)
    np.testing.assert_allclose(sk_qda_means_2, qda_check_2.means_, rtol=1e-5)
    print(f"  {ds_name_2:15s} | LDA/QDA means match")

    print("Assertions passed: sklearn and xorq produce identical results.")

    # Build sklearn composite plot (3 rows x 2 cols)
    fig_sk, axes_sk = plt.subplots(3, 2, figsize=(10, 15))

    # Row 0: isotropic
    result_0 = sk_results[ds_name_0]
    X_0, y_0 = result_0["X"], result_0["y"]
    lda_est_0, qda_est_0 = result_0["lda"], result_0["qda"]
    plot_result(lda_est_0, X_0, y_0, axes_sk[0, 0])
    plot_result(qda_est_0, X_0, y_0, axes_sk[0, 1])
    axes_sk[0, 0].set_ylabel("Isotropic covariance\n", fontsize=10, fontweight="bold")

    # Row 1: shared
    result_1 = sk_results[ds_name_1]
    X_1, y_1 = result_1["X"], result_1["y"]
    lda_est_1, qda_est_1 = result_1["lda"], result_1["qda"]
    plot_result(lda_est_1, X_1, y_1, axes_sk[1, 0])
    plot_result(qda_est_1, X_1, y_1, axes_sk[1, 1])
    axes_sk[1, 0].set_ylabel("Shared covariance\n", fontsize=10, fontweight="bold")

    # Row 2: different
    result_2 = sk_results[ds_name_2]
    X_2, y_2 = result_2["X"], result_2["y"]
    lda_est_2, qda_est_2 = result_2["lda"], result_2["qda"]
    plot_result(lda_est_2, X_2, y_2, axes_sk[2, 0])
    plot_result(qda_est_2, X_2, y_2, axes_sk[2, 1])
    axes_sk[2, 0].set_ylabel("Different covariances\n", fontsize=10, fontweight="bold")

    axes_sk[0, 0].set_title("Linear Discriminant Analysis", fontsize=12)
    axes_sk[0, 1].set_title("Quadratic Discriminant Analysis", fontsize=12)
    plt.suptitle("LDA vs QDA: sklearn", fontsize=14, y=0.995)
    plt.tight_layout()
    out_sk = "imgs/plot_lda_qda_sklearn.png"
    plt.savefig(out_sk, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nsklearn plot saved to {out_sk}")

    # Build xorq composite plot - execute deferred plots in main()
    fig_xo, axes_xo = plt.subplots(3, 2, figsize=(10, 15))

    # Row 0: isotropic
    lda_png_0 = deferred_matplotlib_plot(xo_results[ds_name_0]["table"], _build_lda_plot_isotropic, name="lda_plot").execute()
    qda_png_0 = deferred_matplotlib_plot(xo_results[ds_name_0]["table"], _build_qda_plot_isotropic, name="qda_plot").execute()
    lda_img_0 = load_plot_bytes(lda_png_0)
    qda_img_0 = load_plot_bytes(qda_png_0)
    axes_xo[0, 0].imshow(lda_img_0)
    axes_xo[0, 0].axis("off")
    axes_xo[0, 1].imshow(qda_img_0)
    axes_xo[0, 1].axis("off")
    axes_xo[0, 0].set_ylabel("Isotropic covariance\n", fontsize=10, fontweight="bold")

    # Row 1: shared
    lda_png_1 = deferred_matplotlib_plot(xo_results[ds_name_1]["table"], _build_lda_plot_shared, name="lda_plot").execute()
    qda_png_1 = deferred_matplotlib_plot(xo_results[ds_name_1]["table"], _build_qda_plot_shared, name="qda_plot").execute()
    lda_img_1 = load_plot_bytes(lda_png_1)
    qda_img_1 = load_plot_bytes(qda_png_1)
    axes_xo[1, 0].imshow(lda_img_1)
    axes_xo[1, 0].axis("off")
    axes_xo[1, 1].imshow(qda_img_1)
    axes_xo[1, 1].axis("off")
    axes_xo[1, 0].set_ylabel("Shared covariance\n", fontsize=10, fontweight="bold")

    # Row 2: different
    lda_png_2 = deferred_matplotlib_plot(xo_results[ds_name_2]["table"], _build_lda_plot_different, name="lda_plot").execute()
    qda_png_2 = deferred_matplotlib_plot(xo_results[ds_name_2]["table"], _build_qda_plot_different, name="qda_plot").execute()
    lda_img_2 = load_plot_bytes(lda_png_2)
    qda_img_2 = load_plot_bytes(qda_png_2)
    axes_xo[2, 0].imshow(lda_img_2)
    axes_xo[2, 0].axis("off")
    axes_xo[2, 1].imshow(qda_img_2)
    axes_xo[2, 1].axis("off")
    axes_xo[2, 0].set_ylabel("Different covariances\n", fontsize=10, fontweight="bold")

    axes_xo[0, 0].set_title("Linear Discriminant Analysis", fontsize=12)
    axes_xo[0, 1].set_title("Quadratic Discriminant Analysis", fontsize=12)
    plt.suptitle("LDA vs QDA: xorq", fontsize=14, y=0.995)
    plt.tight_layout()
    out_xo = "imgs/plot_lda_qda_xorq.png"
    plt.savefig(out_xo, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"xorq plot saved to {out_xo}")

    # Composite side-by-side
    sk_img = plt.imread(out_sk)
    xo_img = plt.imread(out_xo)

    fig, axes = plt.subplots(1, 2, figsize=(20, 15))
    axes[0].imshow(sk_img)
    axes[0].set_title("sklearn", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(xo_img)
    axes[1].set_title("xorq", fontsize=14)
    axes[1].axis("off")

    plt.suptitle("Linear and Quadratic Discriminant Analysis: sklearn vs xorq", fontsize=16)
    plt.tight_layout()
    out = "imgs/plot_lda_qda.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Composite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
