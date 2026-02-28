"""SVM Kernels Comparison
========================

sklearn: Train SVC models with different kernel types (linear, poly, rbf, sigmoid)
on a small 2D binary classification dataset. Fit each kernel, generate decision
boundaries via DecisionBoundaryDisplay, plot support vectors.

xorq: Same SVC kernels wrapped in Pipeline.from_instance, fit/predict deferred,
generate deferred decision boundary plots via deferred_matplotlib_plot.

Both produce identical decision boundaries and support vectors.

Dataset: Small synthetic 2D binary classification (16 samples)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from toolz import curry

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    split_data_nop,
)
from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    load_plot_bytes,
    save_fig,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

X_ARRAY = np.array(
    [
        [0.4, -0.7],
        [-1.5, -1.0],
        [-1.4, -0.9],
        [-1.3, -1.2],
        [-1.1, -0.2],
        [-1.2, -0.4],
        [-0.5, 1.2],
        [-1.5, 2.1],
        [1.0, 1.0],
        [1.3, 0.8],
        [1.2, 0.5],
        [0.2, -2.0],
        [0.5, -2.4],
        [0.2, -2.3],
        [0.0, -2.7],
        [1.3, 2.1],
    ]
)
Y_ARRAY = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

KERNELS = ("linear", "poly", "rbf", "sigmoid")
GAMMA = 2
FEATURE_COLS = ("x0", "x1")
TARGET_COL = "y"
PRED_COL = "pred"
X_MIN, X_MAX = -3, 3
Y_MIN, Y_MAX = -3, 3


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Return the fixed 2D binary classification dataset as a DataFrame."""
    return pd.DataFrame(X_ARRAY, columns=list(FEATURE_COLS)).assign(
        **{TARGET_COL: Y_ARRAY}
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_decision_boundary(ax, clf, X, y, kernel, x_min, x_max, y_min, y_max):
    """Plot decision boundary for a fitted SVC classifier."""
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
    common_params = {"estimator": clf, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=150,
        facecolors="none",
        edgecolors="k",
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
    ax.set_title(f"Decision boundary: {kernel} kernel")


@curry
def _build_svc_refit_plot(df, kernel):
    """Refit SVC from materialised DataFrame and build decision boundary plot."""
    X = df[list(FEATURE_COLS)].values
    y = df[TARGET_COL].values
    clf = svm.SVC(kernel=kernel, gamma=GAMMA)
    clf.fit(X, y)
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_decision_boundary(ax, clf, X, y, kernel, X_MIN, X_MAX, Y_MIN, Y_MAX)
    fig.tight_layout()
    return fig


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
        print(f"  {name:8s} accuracy - sklearn: {sk_acc:.4f}, xorq: {xo_acc:.4f}")


def plot_results(comparator):
    X = comparator.df[list(FEATURE_COLS)].values
    y = comparator.df[TARGET_COL].values

    fig, axes = plt.subplots(2, len(KERNELS), figsize=(20, 10))

    for i, kernel in enumerate(KERNELS):
        # Top row: sklearn — result["fitted"] IS the SVC instance (last step)
        clf = comparator.sklearn_results[kernel]["fitted"]
        _plot_decision_boundary(
            axes[0, i], clf, X, y, kernel, X_MIN, X_MAX, Y_MIN, Y_MAX
        )
        if i == 0:
            axes[0, i].set_ylabel("sklearn", fontsize=12, fontweight="bold")

        # Bottom row: xorq — refit inside deferred_matplotlib_plot
        xo_png = deferred_matplotlib_plot(
            xo.memtable(comparator.df),
            _build_svc_refit_plot(kernel=kernel),
        ).execute()
        axes[1, i].imshow(load_plot_bytes(xo_png))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("xorq", fontsize=12, fontweight="bold")

    fig.suptitle("SVM Kernels Comparison: sklearn vs xorq", fontsize=16, y=0.995)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = tuple(KERNELS)
names_pipelines = tuple(
    (kernel, SklearnPipeline([("svc", svm.SVC(kernel=kernel, gamma=GAMMA))]))
    for kernel in KERNELS
)
metrics_names_funcs = (("accuracy", accuracy_score),)

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=TARGET_COL,
    pred=PRED_COL,
    metrics_names_funcs=metrics_names_funcs,
    load_data=load_data,
    split_data=split_data_nop,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)
# expose the exprs to invoke `xorq build plot_svm_kernels.py --expr $expr_name`
(xorq_linear_preds, xorq_poly_preds, xorq_rbf_preds, xorq_sigmoid_preds) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    comparator.result_comparison

    print("\n=== Support Vector Counts ===")
    for kernel in KERNELS:
        clf = comparator.sklearn_results[kernel]["fitted"]
        print(f"  {kernel:8s}: n_support={len(clf.support_vectors_)}")

    save_fig("imgs/plot_svm_kernels.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
