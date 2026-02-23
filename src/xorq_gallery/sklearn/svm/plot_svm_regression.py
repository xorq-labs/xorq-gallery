"""Support Vector Regression
============================

sklearn: Train SVR models with different kernel types (RBF, linear, polynomial)
on synthetic 1D noisy sine data. Fit each kernel, predict, plot fitted curves
and support vectors.

xorq: Same SVR kernels wrapped in Pipeline.from_instance, fit/predict deferred,
generate deferred regression plots via deferred_matplotlib_plot.

Both produce identical predictions and support vectors.

Dataset: Synthetic 1D sine wave with noise (40 samples)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.svm import SVR
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_SAMPLES = 40
KERNELS = ["rbf", "linear", "poly"]
KERNEL_PARAMS = {
    "rbf": {"C": 100, "gamma": 0.1, "epsilon": 0.1},
    "linear": {"C": 100, "gamma": "auto"},
    "poly": {"C": 100, "gamma": "auto", "degree": 3, "epsilon": 0.1, "coef0": 1},
}
KERNEL_LABELS = {"rbf": "RBF", "linear": "Linear", "poly": "Polynomial"}
MODEL_COLORS = {"rbf": "m", "linear": "c", "poly": "g"}


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Generate synthetic 1D sine wave data with noise.

    Matches the sklearn example exactly: 40 samples sorted along x-axis,
    sine wave + noise on every 5th point.
    """
    np.random.seed(RANDOM_SEED)
    X = np.sort(5 * np.random.rand(N_SAMPLES, 1), axis=0)
    y = np.sin(X).ravel()

    # Add noise to every 5th target
    y[::5] += 3 * (0.5 - np.random.rand(8))

    df = pd.DataFrame(X, columns=["x"])
    df["y"] = y
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_svr_result(ax, X, y, svr, kernel, color):
    """Plot SVR fitted curve and support vectors.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    X : numpy.ndarray
        Feature array (N, 1).
    y : numpy.ndarray
        Target array (N,).
    svr : sklearn.svm.SVR
        Fitted SVR model.
    kernel : str
        Kernel type for title.
    color : str
        Color for the fitted line.
    """
    lw = 2

    # Plot fitted curve
    y_pred = svr.predict(X)
    ax.plot(X, y_pred, color=color, lw=lw, label=f"{KERNEL_LABELS[kernel]} model")

    # Plot support vectors
    support_indices = svr.support_
    ax.scatter(
        X[support_indices],
        y[support_indices],
        facecolor="none",
        edgecolor=color,
        s=50,
        label=f"{KERNEL_LABELS[kernel]} support vectors",
    )

    # Plot other training data
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


def _build_plot_function(kernel, X_data, y_data, pipe_template, color):
    """Build a plotting function for deferred_matplotlib_plot.

    Parameters
    ----------
    kernel : str
        Kernel type for title.
    X_data : np.ndarray
        Feature data.
    y_data : np.ndarray
        Target data.
    pipe_template : sklearn.pipeline.Pipeline
        Pipeline to refit for plotting.
    color : str
        Color for the fitted line.

    Returns
    -------
    callable
        Function that takes pred_df and returns a matplotlib figure.
    """
    def _plot(pred_df):
        """Build SVR regression plot from materialized predictions.

        We refit the sklearn model here since we need the fitted estimator
        for support vector access.
        """
        # Refit for plotting - extract the SVR step
        fitted_pipe = pipe_template.fit(X_data, y_data)
        fitted_svr = fitted_pipe.named_steps["svr"]

        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_svr_result(ax, X_data, y_data, fitted_svr, kernel, color)
        plt.tight_layout()
        return fig

    return _plot


# =========================================================================
# SKLEARN WAY -- eager fit and predict
# =========================================================================


def sklearn_way(df):
    """Eager sklearn: fit SVR with each kernel type, predict, collect support vectors.

    Returns dict of kernel -> {svr, predictions, n_support}.
    """
    X = df[["x"]].values
    y = df["y"].values

    results = {}

    # Unroll kernel training (no for loops in sklearn_way/xorq_way)

    # Kernel: rbf
    kernel = "rbf"
    params = KERNEL_PARAMS[kernel]
    svr = SVR(kernel=kernel, **params)
    svr.fit(X, y)
    y_pred = svr.predict(X)
    print(f"  sklearn: kernel={kernel:8s} | n_support={len(svr.support_):2d}")
    results[kernel] = {"svr": svr, "predictions": y_pred, "n_support": len(svr.support_)}

    # Kernel: linear
    kernel = "linear"
    params = KERNEL_PARAMS[kernel]
    svr = SVR(kernel=kernel, **params)
    svr.fit(X, y)
    y_pred = svr.predict(X)
    print(f"  sklearn: kernel={kernel:8s} | n_support={len(svr.support_):2d}")
    results[kernel] = {"svr": svr, "predictions": y_pred, "n_support": len(svr.support_)}

    # Kernel: poly
    kernel = "poly"
    params = KERNEL_PARAMS[kernel]
    svr = SVR(kernel=kernel, **params)
    svr.fit(X, y)
    y_pred = svr.predict(X)
    print(f"  sklearn: kernel={kernel:8s} | n_support={len(svr.support_):2d}")
    results[kernel] = {"svr": svr, "predictions": y_pred, "n_support": len(svr.support_)}

    return results


# =========================================================================
# XORQ WAY -- deferred fit and predict
# =========================================================================


def xorq_way(df):
    """Deferred xorq: wrap SVR in sklearn Pipeline, then Pipeline.from_instance,
    fit deferred, return deferred predictions.

    Returns dict of kernel -> {preds: expr, sklearn_pipe: Pipeline}.
    """
    con = xo.connect()
    table = con.register(df, "svr_data")

    results = {}

    # Unroll kernel training (no for loops in sklearn_way/xorq_way)

    # Kernel: rbf
    kernel = "rbf"
    params = KERNEL_PARAMS[kernel]
    sklearn_pipe = SklearnPipeline([("svr", SVR(kernel=kernel, **params))])
    xorq_pipe = Pipeline.from_instance(sklearn_pipe)
    fitted = xorq_pipe.fit(table, features=("x",), target="y")
    preds = fitted.predict(table, name="pred")
    results[kernel] = {"preds": preds, "sklearn_pipe": sklearn_pipe}

    # Kernel: linear
    kernel = "linear"
    params = KERNEL_PARAMS[kernel]
    sklearn_pipe = SklearnPipeline([("svr", SVR(kernel=kernel, **params))])
    xorq_pipe = Pipeline.from_instance(sklearn_pipe)
    fitted = xorq_pipe.fit(table, features=("x",), target="y")
    preds = fitted.predict(table, name="pred")
    results[kernel] = {"preds": preds, "sklearn_pipe": sklearn_pipe}

    # Kernel: poly
    kernel = "poly"
    params = KERNEL_PARAMS[kernel]
    sklearn_pipe = SklearnPipeline([("svr", SVR(kernel=kernel, **params))])
    xorq_pipe = Pipeline.from_instance(sklearn_pipe)
    fitted = xorq_pipe.fit(table, features=("x",), target="y")
    preds = fitted.predict(table, name="pred")
    results[kernel] = {"preds": preds, "sklearn_pipe": sklearn_pipe}

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(df)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(df)

    # Execute deferred predictions and verify they match
    print("\n=== ASSERTIONS ===")
    X = df[["x"]].values
    y = df["y"].values

    # Kernel: rbf
    kernel = "rbf"
    xo_preds_df = xo_results[kernel]["preds"].execute()
    sk_preds = sk_results[kernel]["predictions"]
    xo_preds = xo_preds_df["pred"].values
    np.testing.assert_allclose(sk_preds, xo_preds, rtol=1e-10)
    print(f"  xorq:   kernel={kernel:8s} | predictions match sklearn")

    # Kernel: linear
    kernel = "linear"
    xo_preds_df = xo_results[kernel]["preds"].execute()
    sk_preds = sk_results[kernel]["predictions"]
    xo_preds = xo_preds_df["pred"].values
    np.testing.assert_allclose(sk_preds, xo_preds, rtol=1e-10)
    print(f"  xorq:   kernel={kernel:8s} | predictions match sklearn")

    # Kernel: poly
    kernel = "poly"
    xo_preds_df = xo_results[kernel]["preds"].execute()
    sk_preds = sk_results[kernel]["predictions"]
    xo_preds = xo_preds_df["pred"].values
    np.testing.assert_allclose(sk_preds, xo_preds, rtol=1e-10)
    print(f"  xorq:   kernel={kernel:8s} | predictions match sklearn")

    print("Assertions passed: sklearn and xorq predictions match.")

    # Generate deferred plots (deferred_matplotlib_plot ONLY in main())
    xo_plots = {}

    # Kernel: rbf
    kernel = "rbf"
    plot_fn = _build_plot_function(kernel, X, y, xo_results[kernel]["sklearn_pipe"], MODEL_COLORS[kernel])
    xo_plots[kernel] = deferred_matplotlib_plot(xo_results[kernel]["preds"], plot_fn).execute()

    # Kernel: linear
    kernel = "linear"
    plot_fn = _build_plot_function(kernel, X, y, xo_results[kernel]["sklearn_pipe"], MODEL_COLORS[kernel])
    xo_plots[kernel] = deferred_matplotlib_plot(xo_results[kernel]["preds"], plot_fn).execute()

    # Kernel: poly
    kernel = "poly"
    plot_fn = _build_plot_function(kernel, X, y, xo_results[kernel]["sklearn_pipe"], MODEL_COLORS[kernel])
    xo_plots[kernel] = deferred_matplotlib_plot(xo_results[kernel]["preds"], plot_fn).execute()

    # Build composite plot: 2 rows x 3 cols (one row per approach)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: sklearn results
    # Kernel: rbf
    ax = axes[0, 0]
    kernel = "rbf"
    svr = sk_results[kernel]["svr"]
    color = MODEL_COLORS[kernel]
    _plot_svr_result(ax, X, y, svr, kernel, color)
    ax.text(-0.3, 0.5, "sklearn", transform=ax.transAxes, fontsize=12, fontweight="bold", va="center", rotation=90)

    # Kernel: linear
    ax = axes[0, 1]
    kernel = "linear"
    svr = sk_results[kernel]["svr"]
    color = MODEL_COLORS[kernel]
    _plot_svr_result(ax, X, y, svr, kernel, color)

    # Kernel: poly
    ax = axes[0, 2]
    kernel = "poly"
    svr = sk_results[kernel]["svr"]
    color = MODEL_COLORS[kernel]
    _plot_svr_result(ax, X, y, svr, kernel, color)

    # Bottom row: xorq results (load deferred plots)
    # Kernel: rbf
    ax = axes[1, 0]
    kernel = "rbf"
    img = load_plot_bytes(xo_plots[kernel])
    ax.imshow(img)
    ax.axis("off")
    ax.text(-0.3, 0.5, "xorq", transform=ax.transAxes, fontsize=12, fontweight="bold", va="center", rotation=90)

    # Kernel: linear
    ax = axes[1, 1]
    kernel = "linear"
    img = load_plot_bytes(xo_plots[kernel])
    ax.imshow(img)
    ax.axis("off")

    # Kernel: poly
    ax = axes[1, 2]
    kernel = "poly"
    img = load_plot_bytes(xo_plots[kernel])
    ax.imshow(img)
    ax.axis("off")

    plt.suptitle(
        "Support Vector Regression: sklearn vs xorq", fontsize=16, y=0.995
    )
    plt.tight_layout()
    out = "imgs/svm_regression.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
