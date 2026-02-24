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
from toolz import curry
from xorq.expr.ml.pipeline_lib import Pipeline
from xorq.ibis_yaml.utils import freeze

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_SAMPLES = 40
KERNELS = ("rbf", "linear", "poly")
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


@curry
def _build_svr_plot(pred_df, kernel, X_data, y_data, pipe_template, color):
    """Build SVR regression plot from materialized predictions.

    X_data and y_data are passed as tuples (for xorq hashability) and
    converted back to numpy arrays inside this function.
    """
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    fitted_pipe = pipe_template.fit(X_data, y_data)
    fitted_svr = fitted_pipe.named_steps["svr"]
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_svr_result(ax, X_data, y_data, fitted_svr, kernel, color)
    fig.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit and predict
# =========================================================================


def sklearn_way(df):
    """Eager sklearn: fit SVR with each kernel type, predict, collect support vectors."""
    X = df[["x"]].values
    y = df["y"].values

    results = {}
    for kernel in KERNELS:
        params = KERNEL_PARAMS[kernel]
        svr = SVR(kernel=kernel, **params)
        svr.fit(X, y)
        y_pred = svr.predict(X)
        print(f"  sklearn: kernel={kernel:8s} | n_support={len(svr.support_):2d}")
        results[kernel] = {
            "svr": svr,
            "predictions": y_pred,
            "n_support": len(svr.support_),
        }

    return results


# =========================================================================
# XORQ WAY -- deferred fit and predict
# =========================================================================


def xorq_way(df):
    """Deferred xorq: wrap SVR in Pipeline.from_instance, fit/predict deferred."""
    con = xo.connect()
    table = con.register(df, "svr_data")

    results = {}
    for kernel in KERNELS:
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

    for kernel in KERNELS:
        xo_preds_df = xo_results[kernel]["preds"].execute()
        sk_preds = sk_results[kernel]["predictions"]
        xo_preds = xo_preds_df["pred"].values
        np.testing.assert_allclose(sk_preds, xo_preds, rtol=1e-10)
        print(f"  xorq:   kernel={kernel:8s} | predictions match sklearn")

    print("Assertions passed: sklearn and xorq predictions match.")

    # Generate deferred plots
    xo_plots = {}
    for kernel in KERNELS:
        plot_fn = _build_svr_plot(
            kernel=kernel,
            X_data=freeze(X.tolist()),
            y_data=freeze(y.tolist()),
            pipe_template=xo_results[kernel]["sklearn_pipe"],
            color=MODEL_COLORS[kernel],
        )
        xo_plots[kernel] = deferred_matplotlib_plot(
            xo_results[kernel]["preds"], plot_fn
        ).execute()

    # Build composite plot: 2 rows x 3 cols
    fig, axes = plt.subplots(2, len(KERNELS), figsize=(15, 10))

    for idx, kernel in enumerate(KERNELS):
        # Top row: sklearn
        svr = sk_results[kernel]["svr"]
        _plot_svr_result(axes[0, idx], X, y, svr, kernel, MODEL_COLORS[kernel])
        if idx == 0:
            axes[0, idx].text(
                -0.3,
                0.5,
                "sklearn",
                transform=axes[0, idx].transAxes,
                fontsize=12,
                fontweight="bold",
                va="center",
                rotation=90,
            )

        # Bottom row: xorq
        img = load_plot_bytes(xo_plots[kernel])
        axes[1, idx].imshow(img)
        axes[1, idx].axis("off")
        if idx == 0:
            axes[1, idx].text(
                -0.3,
                0.5,
                "xorq",
                transform=axes[1, idx].transAxes,
                fontsize=12,
                fontweight="bold",
                va="center",
                rotation=90,
            )

    fig.suptitle("Support Vector Regression: sklearn vs xorq", fontsize=16, y=0.995)
    fig.tight_layout()
    out = "imgs/svm_regression.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
