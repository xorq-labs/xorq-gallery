"""Linear and Quadratic Discriminant Analysis
===============================================

sklearn: Generate three synthetic 2D binary classification datasets with different
covariance structures (isotropic, shared, varying), fit LDA and QDA classifiers on
each, plot decision boundaries with covariance ellipsoids at 2 standard deviations.

xorq: Same LDA/QDA classifiers wrapped in Pipeline.from_instance, fit/predict
deferred, generate deferred decision boundary plots with covariance ellipsoids.

Both produce identical decision boundaries and covariance estimates.

Dataset: Synthetic 2D Gaussian blobs with varying covariance matrices

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/classification/plot_lda_qda.py
"""

from __future__ import annotations

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
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline as SklearnPipeline

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    split_data_nop,
)
from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
    save_fig,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 0
N_SAMPLES = 300
N_SAMPLES_ISOTROPIC = 1_000

FEATURE_COLS = ("x0", "x1")
TARGET_COL = "y"
PRED_COL = "pred"

dataset_names = ("isotropic", "shared", "different")


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _make_data(n_samples, cov_class_1, cov_class_2, seed=0):
    """Generate synthetic binary classification data with specified covariance."""
    rng = np.random.RandomState(seed)
    X = np.concatenate(
        [
            rng.randn(n_samples, 2) @ cov_class_1,
            rng.randn(n_samples, 2) @ cov_class_2 + np.array([1, 1]),
        ]
    )
    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    return pd.DataFrame(X, columns=list(FEATURE_COLS)).assign(**{TARGET_COL: y})


def load_data_isotropic():
    cov = np.array([[1, 0], [0, 1]])
    return _make_data(N_SAMPLES_ISOTROPIC, cov, cov, seed=RANDOM_STATE)


def load_data_shared():
    cov = np.array([[0.0, -0.23], [0.83, 0.23]])
    return _make_data(N_SAMPLES, cov, cov, seed=RANDOM_STATE)


def load_data_different():
    cov1 = np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0
    return _make_data(N_SAMPLES, cov1, cov1.T, seed=RANDOM_STATE)


_LOAD_DATA_FNS = {
    "isotropic": load_data_isotropic,
    "shared": load_data_shared,
    "different": load_data_different,
}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_ellipse(mean, cov, color, ax):
    """Plot covariance ellipse at 2 standard deviations."""
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = 180 * np.arctan(u[1] / u[0]) / np.pi
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


def _plot_result(estimator, X, y, ax):
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
        X_wrong[:, 0], X_wrong[:, 1], c=y_wrong, s=30, cmap=cmap, alpha=0.9, marker="x"
    )
    ax.scatter(
        estimator.means_[:, 0],
        estimator.means_[:, 1],
        c="yellow",
        s=200,
        marker="*",
        edgecolor="black",
    )
    covariance = (
        [estimator.covariance_] * 2
        if isinstance(estimator, LinearDiscriminantAnalysis)
        else estimator.covariance_
    )
    _plot_ellipse(estimator.means_[0], covariance[0], "tab:red", ax)
    _plot_ellipse(estimator.means_[1], covariance[1], "tab:blue", ax)
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set(xticks=[], yticks=[])


def _build_lda_plot(df):
    """Refit LDA and build decision boundary plot from materialised DataFrame."""
    X = df[list(FEATURE_COLS)].values
    y = df[TARGET_COL].values
    estimator = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    estimator.fit(X, y)
    fig, ax = plt.subplots(figsize=(5, 5))
    _plot_result(estimator, X, y, ax)
    ax.set_title(LDA_NAME)
    fig.tight_layout()
    return fig


def _build_qda_plot(df):
    """Refit QDA and build decision boundary plot from materialised DataFrame."""
    X = df[list(FEATURE_COLS)].values
    y = df[TARGET_COL].values
    estimator = QuadraticDiscriminantAnalysis(store_covariance=True)
    estimator.fit(X, y)
    fig, ax = plt.subplots(figsize=(5, 5))
    _plot_result(estimator, X, y, ax)
    ax.set_title(QDA_NAME)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Comparator callbacks (shared by all 3 dataset comparators)
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

    fig_sk, axes_sk = plt.subplots(1, 2, figsize=(10, 5))
    for col, name in enumerate([LDA_NAME, QDA_NAME]):
        # result["fitted"] is the last pipeline step (LDA or QDA estimator directly)
        estimator = comparator.sklearn_results[name]["fitted"]
        _plot_result(estimator, X, y, axes_sk[col])
        axes_sk[col].set_title(name)
    fig_sk.tight_layout()

    fig_xo, axes_xo = plt.subplots(1, 2, figsize=(10, 5))
    for col, plot_fn in enumerate([_build_lda_plot, _build_qda_plot]):
        xo_png = deferred_matplotlib_plot(xo.memtable(comparator.df), plot_fn).execute()
        axes_xo[col].imshow(load_plot_bytes(xo_png))
        axes_xo[col].axis("off")
    fig_xo.tight_layout()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(fig_to_image(fig_sk))
    axes[0].set_title("sklearn")
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(fig_xo))
    axes[1].set_title("xorq")
    axes[1].axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup: one comparator per dataset
# ---------------------------------------------------------------------------

methods = (LDA_NAME, QDA_NAME) = ("LDA", "QDA")
names_pipelines = (
    (
        LDA_NAME,
        SklearnPipeline(
            [("lda", LinearDiscriminantAnalysis(solver="svd", store_covariance=True))]
        ),
    ),
    (
        QDA_NAME,
        SklearnPipeline(
            [("qda", QuadraticDiscriminantAnalysis(store_covariance=True))]
        ),
    ),
)
metrics_names_funcs = (("accuracy", accuracy_score),)

comparators = {
    ds_name: SklearnXorqComparator(
        names_pipelines=names_pipelines,
        features=FEATURE_COLS,
        target=TARGET_COL,
        pred=PRED_COL,
        metrics_names_funcs=metrics_names_funcs,
        load_data=_LOAD_DATA_FNS[ds_name],
        split_data=split_data_nop,
        compare_results_fn=compare_results,
        plot_results_fn=plot_results,
    )
    for ds_name in dataset_names
}
# expose the exprs to invoke `xorq build plot_lda_qda.py --expr $expr_name`
(
    xorq_lda_iso_preds,
    xorq_lda_shared_preds,
    xorq_lda_diff_preds,
    xorq_qda_iso_preds,
    xorq_qda_shared_preds,
    xorq_qda_diff_preds,
) = (
    comparators[ds_name].deferred_xorq_results[clf_name]["preds"]
    for clf_name in methods
    for ds_name in dataset_names
)


def main():
    row_labels = (
        "Isotropic covariance\n(LDA = QDA: same covariance per class)",
        "Shared covariance\n(LDA = QDA: same covariance per class)",
        "Different covariances\n(LDA != QDA: per-class covariance matters)",
    )
    for ds_name in dataset_names:
        comparators[ds_name].result_comparison

    row_figs = [comparators[ds_name].plot_results() for ds_name in dataset_names]

    fig, axes = plt.subplots(len(dataset_names), 1, figsize=(16, 20))
    for row, (row_fig, label) in enumerate(zip(row_figs, row_labels)):
        axes[row].imshow(fig_to_image(row_fig))
        axes[row].set_title(label, fontsize=12, fontstyle="italic", pad=8)
        axes[row].axis("off")
    fig.suptitle(
        "Linear and Quadratic Discriminant Analysis: sklearn vs xorq", fontsize=16
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig("imgs/plot_lda_qda.png", fig)


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
