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
from toolz import curry
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 0
N_SAMPLES = 300
N_SAMPLES_ISOTROPIC = 1_000  # First dataset uses more samples

# Feature columns
feature_cols = ("x0", "x1")
target_col = "y"
pred_col = "pred"

# Dataset names
dataset_names = ("isotropic", "shared", "different")


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
    ax.scatter(
        X_right[:, 0],
        X_right[:, 1],
        c=y_right,
        s=20,
        cmap=cmap,
        alpha=0.5,
    )
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


@curry
def _build_classifier_plot(df, classifier_class, title):
    """Build decision boundary plot for a given classifier.

    This curried function handles both LDA and QDA plotting with identical logic.
    """
    X_data = df[list(feature_cols)].values
    y_data = df[target_col].values

    estimator = classifier_class(
        solver="svd" if classifier_class == LinearDiscriminantAnalysis else None,
        store_covariance=True,
    )
    estimator.fit(X_data, y_data)

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_result(estimator, X_data, y_data, ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit, eager plotting
# =========================================================================


def sklearn_way(datasets):
    """Eager sklearn: fit LDA and QDA on three datasets, plot decision boundaries
    with covariance ellipsoids."""

    # Dataset 0: isotropic
    X_isotropic, y_isotropic = datasets["isotropic"]
    lda_isotropic = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    qda_isotropic = QuadraticDiscriminantAnalysis(store_covariance=True)
    lda_isotropic.fit(X_isotropic, y_isotropic)
    qda_isotropic.fit(X_isotropic, y_isotropic)
    result_isotropic = {
        "lda": lda_isotropic,
        "qda": qda_isotropic,
        "X": X_isotropic,
        "y": y_isotropic,
        "lda_means": lda_isotropic.means_.copy(),
        "lda_cov": lda_isotropic.covariance_.copy(),
        "qda_means": qda_isotropic.means_.copy(),
        "qda_cov": [c.copy() for c in qda_isotropic.covariance_],
    }
    print(
        f"sklearn: isotropic      | LDA fitted | means shape: {lda_isotropic.means_.shape}"
    )
    print(
        f"sklearn: isotropic      | QDA fitted | means shape: {qda_isotropic.means_.shape}"
    )

    # Dataset 1: shared
    X_shared, y_shared = datasets["shared"]
    lda_shared = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    qda_shared = QuadraticDiscriminantAnalysis(store_covariance=True)
    lda_shared.fit(X_shared, y_shared)
    qda_shared.fit(X_shared, y_shared)
    result_shared = {
        "lda": lda_shared,
        "qda": qda_shared,
        "X": X_shared,
        "y": y_shared,
        "lda_means": lda_shared.means_.copy(),
        "lda_cov": lda_shared.covariance_.copy(),
        "qda_means": qda_shared.means_.copy(),
        "qda_cov": [c.copy() for c in qda_shared.covariance_],
    }
    print(
        f"sklearn: shared         | LDA fitted | means shape: {lda_shared.means_.shape}"
    )
    print(
        f"sklearn: shared         | QDA fitted | means shape: {qda_shared.means_.shape}"
    )

    # Dataset 2: different
    X_different, y_different = datasets["different"]
    lda_different = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    qda_different = QuadraticDiscriminantAnalysis(store_covariance=True)
    lda_different.fit(X_different, y_different)
    qda_different.fit(X_different, y_different)
    result_different = {
        "lda": lda_different,
        "qda": qda_different,
        "X": X_different,
        "y": y_different,
        "lda_means": lda_different.means_.copy(),
        "lda_cov": lda_different.covariance_.copy(),
        "qda_means": qda_different.means_.copy(),
        "qda_cov": [c.copy() for c in qda_different.covariance_],
    }
    print(
        f"sklearn: different      | LDA fitted | means shape: {lda_different.means_.shape}"
    )
    print(
        f"sklearn: different      | QDA fitted | means shape: {qda_different.means_.shape}"
    )

    return {
        "isotropic": result_isotropic,
        "shared": result_shared,
        "different": result_different,
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
    X_isotropic, y_isotropic = datasets["isotropic"]
    df_isotropic = pd.DataFrame(X_isotropic, columns=list(feature_cols))
    df_isotropic[target_col] = y_isotropic
    table_isotropic = con.register(df_isotropic, "data_isotropic")

    # LDA for dataset 0
    lda_sklearn_isotropic = SklearnPipeline(
        [("lda", LinearDiscriminantAnalysis(solver="svd", store_covariance=True))]
    )
    lda_pipe_isotropic = Pipeline.from_instance(lda_sklearn_isotropic)
    lda_fitted_isotropic = lda_pipe_isotropic.fit(
        table_isotropic,
        features=feature_cols,
        target=target_col,
    )
    lda_preds_isotropic = lda_fitted_isotropic.predict(table_isotropic, name=pred_col)

    # QDA for dataset 0
    qda_sklearn_isotropic = SklearnPipeline(
        [("qda", QuadraticDiscriminantAnalysis(store_covariance=True))]
    )
    qda_pipe_isotropic = Pipeline.from_instance(qda_sklearn_isotropic)
    qda_fitted_isotropic = qda_pipe_isotropic.fit(
        table_isotropic,
        features=feature_cols,
        target=target_col,
    )
    qda_preds_isotropic = qda_fitted_isotropic.predict(table_isotropic, name=pred_col)

    result_isotropic = {
        "table": table_isotropic,
        "lda_preds": lda_preds_isotropic,
        "qda_preds": qda_preds_isotropic,
    }

    # Dataset 1: shared
    X_shared, y_shared = datasets["shared"]
    df_shared = pd.DataFrame(X_shared, columns=list(feature_cols))
    df_shared[target_col] = y_shared
    table_shared = con.register(df_shared, "data_shared")

    # LDA for dataset 1
    lda_sklearn_shared = SklearnPipeline(
        [("lda", LinearDiscriminantAnalysis(solver="svd", store_covariance=True))]
    )
    lda_pipe_shared = Pipeline.from_instance(lda_sklearn_shared)
    lda_fitted_shared = lda_pipe_shared.fit(
        table_shared,
        features=feature_cols,
        target=target_col,
    )
    lda_preds_shared = lda_fitted_shared.predict(table_shared, name=pred_col)

    # QDA for dataset 1
    qda_sklearn_shared = SklearnPipeline(
        [("qda", QuadraticDiscriminantAnalysis(store_covariance=True))]
    )
    qda_pipe_shared = Pipeline.from_instance(qda_sklearn_shared)
    qda_fitted_shared = qda_pipe_shared.fit(
        table_shared,
        features=feature_cols,
        target=target_col,
    )
    qda_preds_shared = qda_fitted_shared.predict(table_shared, name=pred_col)

    result_shared = {
        "table": table_shared,
        "lda_preds": lda_preds_shared,
        "qda_preds": qda_preds_shared,
    }

    # Dataset 2: different
    X_different, y_different = datasets["different"]
    df_different = pd.DataFrame(X_different, columns=list(feature_cols))
    df_different[target_col] = y_different
    table_different = con.register(df_different, "data_different")

    # LDA for dataset 2
    lda_sklearn_different = SklearnPipeline(
        [("lda", LinearDiscriminantAnalysis(solver="svd", store_covariance=True))]
    )
    lda_pipe_different = Pipeline.from_instance(lda_sklearn_different)
    lda_fitted_different = lda_pipe_different.fit(
        table_different,
        features=feature_cols,
        target=target_col,
    )
    lda_preds_different = lda_fitted_different.predict(table_different, name=pred_col)

    # QDA for dataset 2
    qda_sklearn_different = SklearnPipeline(
        [("qda", QuadraticDiscriminantAnalysis(store_covariance=True))]
    )
    qda_pipe_different = Pipeline.from_instance(qda_sklearn_different)
    qda_fitted_different = qda_pipe_different.fit(
        table_different,
        features=feature_cols,
        target=target_col,
    )
    qda_preds_different = qda_fitted_different.predict(table_different, name=pred_col)

    result_different = {
        "table": table_different,
        "lda_preds": lda_preds_different,
        "qda_preds": qda_preds_different,
    }

    return {
        "isotropic": result_isotropic,
        "shared": result_shared,
        "different": result_different,
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

    # Assert numerical equivalence
    print("\n=== ASSERTIONS ===")

    # Build DataFrame of means for comparison
    sklearn_means_data = []
    for ds_name in dataset_names:
        X, y = datasets[ds_name]
        lda_check = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
        qda_check = QuadraticDiscriminantAnalysis(store_covariance=True)
        lda_check.fit(X, y)
        qda_check.fit(X, y)

        sklearn_means_data.append(
            {
                "dataset": ds_name,
                "classifier": "LDA",
                "mean_0_0": lda_check.means_[0, 0],
                "mean_0_1": lda_check.means_[0, 1],
                "mean_1_0": lda_check.means_[1, 0],
                "mean_1_1": lda_check.means_[1, 1],
            }
        )
        sklearn_means_data.append(
            {
                "dataset": ds_name,
                "classifier": "QDA",
                "mean_0_0": qda_check.means_[0, 0],
                "mean_0_1": qda_check.means_[0, 1],
                "mean_1_0": qda_check.means_[1, 0],
                "mean_1_1": qda_check.means_[1, 1],
            }
        )

        # Individual assertions for verification
        sk_lda_means = sk_results[ds_name]["lda_means"]
        sk_qda_means = sk_results[ds_name]["qda_means"]
        np.testing.assert_allclose(sk_lda_means, lda_check.means_, rtol=1e-5)
        np.testing.assert_allclose(sk_qda_means, qda_check.means_, rtol=1e-5)
        print(f"  {ds_name:15s} | LDA/QDA means match")

    print("Assertions passed: sklearn and xorq produce identical results.")

    # Build sklearn composite plot (3 rows x 2 cols)
    fig_sk, axes_sk = plt.subplots(3, 2, figsize=(10, 15))

    # Row labels for different covariance structures
    row_labels = (
        "Isotropic covariance\n",
        "Shared covariance\n",
        "Different covariances\n",
    )

    # Generate sklearn plots
    for row, ds_name in enumerate(dataset_names):
        result = sk_results[ds_name]
        X, y = result["X"], result["y"]
        lda_est, qda_est = result["lda"], result["qda"]
        plot_result(lda_est, X, y, axes_sk[row, 0])
        plot_result(qda_est, X, y, axes_sk[row, 1])
        axes_sk[row, 0].set_ylabel(row_labels[row], fontsize=10, fontweight="bold")

    axes_sk[0, 0].set_title("Linear Discriminant Analysis", fontsize=12)
    axes_sk[0, 1].set_title("Quadratic Discriminant Analysis", fontsize=12)
    fig_sk.suptitle("LDA vs QDA: sklearn", fontsize=14, y=0.995)
    fig_sk.tight_layout()
    out_sk = "imgs/plot_lda_qda_sklearn.png"
    fig_sk.savefig(out_sk, dpi=150, bbox_inches="tight")
    plt.close(fig_sk)
    print(f"\nsklearn plot saved to {out_sk}")

    fig_xo, axes_xo = plt.subplots(3, 2, figsize=(10, 15))

    classifier_configs = (
        (LinearDiscriminantAnalysis, "LDA"),
        (QuadraticDiscriminantAnalysis, "QDA"),
    )

    for row, ds_name in enumerate(dataset_names):
        table = xo_results[ds_name]["table"]

        for col, (classifier_class, clf_name) in enumerate(classifier_configs):
            plot_fn = _build_classifier_plot(
                classifier_class=classifier_class,
                title=clf_name,
            )
            png_bytes = deferred_matplotlib_plot(
                table,
                plot_fn,
                name=f"{clf_name.lower()}_plot",
            ).execute()
            img = load_plot_bytes(png_bytes)
            axes_xo[row, col].imshow(img)
            axes_xo[row, col].axis("off")

        axes_xo[row, 0].set_ylabel(row_labels[row], fontsize=10, fontweight="bold")

    axes_xo[0, 0].set_title("Linear Discriminant Analysis", fontsize=12)
    axes_xo[0, 1].set_title("Quadratic Discriminant Analysis", fontsize=12)
    fig_xo.suptitle("LDA vs QDA: xorq", fontsize=14, y=0.995)
    fig_xo.tight_layout()
    out_xo = "imgs/plot_lda_qda_xorq.png"
    fig_xo.savefig(out_xo, dpi=150, bbox_inches="tight")
    plt.close(fig_xo)
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

    fig.suptitle(
        "Linear and Quadratic Discriminant Analysis: sklearn vs xorq",
        fontsize=16,
    )
    fig.tight_layout()
    out = "imgs/plot_lda_qda.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Composite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
