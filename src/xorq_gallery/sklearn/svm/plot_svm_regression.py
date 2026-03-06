"""Support Vector Regression
============================

sklearn: Train SVR models with different kernel types (RBF, linear, polynomial)
on synthetic 1D noisy sine data. Fit each kernel, predict, plot fitted curves
and support vectors.

xorq: Same SVR kernels wrapped in Pipeline.from_instance, fit/predict deferred,
regression plots drawn from the xorq-fitted model.

Both produce identical predictions and support vectors.

Dataset: Synthetic 1D sine wave with noise (40 samples)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/svm/plot_svm_regression.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.svm import SVR

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    split_data_nop,
)
from xorq_gallery.utils import save_fig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_SAMPLES = 40
FEATURE_COLS = ("x",)
TARGET_COL = "y"
PRED_COL = "pred"

KERNELS = ("rbf", "linear", "poly")
KERNEL_PARAMS = {
    "rbf": {"C": 100, "gamma": 0.1, "epsilon": 0.1},
    "linear": {"C": 100, "gamma": "auto"},
    "poly": {"C": 100, "gamma": "auto", "degree": 3, "epsilon": 0.1, "coef0": 1},
}
KERNEL_LABELS = {"rbf": "RBF", "linear": "Linear", "poly": "Polynomial"}
MODEL_COLORS = {"rbf": "m", "linear": "c", "poly": "g"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Generate synthetic 1D sine wave data with noise."""
    np.random.seed(RANDOM_SEED)
    X = np.sort(5 * np.random.rand(N_SAMPLES, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - np.random.rand(8))
    return pd.DataFrame({FEATURE_COLS[0]: X[:, 0], TARGET_COL: y})


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_svr_result(ax, X, y, svr, kernel, color):
    """Plot SVR fitted curve and support vectors."""
    lw = 2
    y_pred = svr.predict(X)
    ax.plot(X, y_pred, color=color, lw=lw, label=f"{KERNEL_LABELS[kernel]} model")
    support_indices = svr.support_
    ax.scatter(
        X[support_indices],
        y[support_indices],
        facecolor="none",
        edgecolor=color,
        s=50,
        label=f"{KERNEL_LABELS[kernel]} support vectors",
    )
    other_indices = np.setdiff1d(np.arange(len(X)), support_indices)
    ax.scatter(
        X[other_indices],
        y[other_indices],
        facecolor="none",
        edgecolor="k",
        s=50,
        label="other training data",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=1,
        fancybox=True,
        shadow=True,
    )
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title(f"{KERNEL_LABELS[kernel]} kernel")


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
        sk_r2 = sklearn_result["metrics"]["r2"]
        xo_r2 = xorq_result["metrics"]["r2"]
        print(f"  {name:8s} r2 - sklearn: {sk_r2:.4f}, xorq: {xo_r2:.4f}")


def plot_results(comparator):
    X = comparator.df[[FEATURE_COLS[0]]].values
    y = comparator.df[TARGET_COL].values

    fig, axes = plt.subplots(2, len(KERNELS), figsize=(15, 10))

    for idx, kernel in enumerate(KERNELS):
        # Top row: sklearn
        sk_svr = comparator.sklearn_results[kernel]["fitted"]
        _plot_svr_result(axes[0, idx], X, y, sk_svr, kernel, MODEL_COLORS[kernel])
        if idx == 0:
            axes[0, idx].set_ylabel("sklearn", fontsize=12, fontweight="bold")

        # Bottom row: xorq (fitted model is the same sklearn estimator)
        xo_svr = comparator.xorq_results[kernel]["fitted"]
        _plot_svr_result(axes[1, idx], X, y, xo_svr, kernel, MODEL_COLORS[kernel])
        if idx == 0:
            axes[1, idx].set_ylabel("xorq", fontsize=12, fontweight="bold")

    fig.suptitle("Support Vector Regression: sklearn vs xorq", fontsize=16, y=0.995)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = tuple(KERNELS)
names_pipelines = tuple(
    (kernel, SklearnPipeline([("svr", SVR(kernel=kernel, **KERNEL_PARAMS[kernel]))]))
    for kernel in KERNELS
)
metrics_names_funcs = (("r2", r2_score),)

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
# expose the exprs to invoke `xorq build plot_svm_regression.py --expr $expr_name`
(xorq_rbf_preds, xorq_linear_preds, xorq_poly_preds) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    comparator.result_comparison

    print("\n=== Support Vector Counts ===")
    for kernel in KERNELS:
        svr = comparator.sklearn_results[kernel]["fitted"]
        print(f"  {kernel:8s}: n_support={len(svr.support_)}")

    save_fig("imgs/plot_svm_regression.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
