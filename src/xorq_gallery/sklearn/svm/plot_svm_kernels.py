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

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import Pipeline as SklearnPipeline
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

# Fixed dataset from sklearn example
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
X_MIN, X_MAX = -3, 3
Y_MIN, Y_MAX = -3, 3


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Return the fixed 2D binary classification dataset as a pandas DataFrame."""
    df = pd.DataFrame(X_ARRAY, columns=["x0", "x1"])
    df["y"] = Y_ARRAY
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_decision_boundary(ax, clf, X, y, kernel, x_min, x_max, y_min, y_max):
    """Plot decision boundary for a fitted SVC classifier.

    Follows the sklearn example pattern: pcolormesh for predicted regions,
    contour for decision function levels, scatter for support vectors and data.
    """
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundary and margins
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

    # Plot support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=150,
        facecolors="none",
        edgecolors="k",
    )

    # Plot data points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
    ax.set_title(f"Decision boundary: {kernel} kernel")


@curry
def _build_svm_kernel_plot(pred_df, kernel, X_data, y_data, pipe_template):
    """Build decision boundary plot from materialized predictions.

    X_data and y_data are passed as tuples (for xorq hashability) and
    converted back to numpy arrays inside this function.
    """
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    fitted_pipe = pipe_template.fit(X_data, y_data)
    fitted_clf = fitted_pipe.named_steps["svc"]
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_decision_boundary(
        ax, fitted_clf, X_data, y_data, kernel, X_MIN, X_MAX, Y_MIN, Y_MAX
    )
    fig.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit, decision boundary plots
# =========================================================================


def sklearn_way(df):
    """Eager sklearn: fit SVC with each kernel type, generate decision boundary
    plots, collect support vectors.

    Returns dict of kernel -> {clf, support_vectors, n_support}.
    """
    X = df[["x0", "x1"]].values
    y = df["y"].values

    results = {}
    for kernel in KERNELS:
        clf = svm.SVC(kernel=kernel, gamma=GAMMA)
        clf.fit(X, y)
        print(
            f"  sklearn: kernel={kernel:8s} | n_support={len(clf.support_vectors_):2d}"
        )
        results[kernel] = {
            "clf": clf,
            "support_vectors": clf.support_vectors_,
            "n_support": len(clf.support_vectors_),
        }

    return results


# =========================================================================
# XORQ WAY -- deferred fit, deferred decision boundary plots
# =========================================================================


def xorq_way(df):
    """Deferred xorq: wrap SVC in sklearn Pipeline, then Pipeline.from_instance,
    fit deferred, return deferred predictions.

    Returns dict of kernel -> {preds: expr}.
    """
    con = xo.connect()
    table = con.register(df, "svm_data")

    results = {}
    for kernel in KERNELS:
        sklearn_pipe = SklearnPipeline([("svc", svm.SVC(kernel=kernel, gamma=GAMMA))])
        xorq_pipe = Pipeline.from_instance(sklearn_pipe)
        fitted = xorq_pipe.fit(table, features=("x0", "x1"), target="y")
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

    # Execute deferred predictions and verify predictions match
    print("\n=== ASSERTIONS ===")
    X = df[["x0", "x1"]].values
    y = df["y"].values

    for kernel in KERNELS:
        xo_preds_df = xo_results[kernel]["preds"].execute()
        sk_preds = sk_results[kernel]["clf"].predict(X)
        xo_preds = xo_preds_df["pred"].values
        np.testing.assert_array_equal(sk_preds, xo_preds)
        print(f"  xorq:   kernel={kernel:8s} | predictions match sklearn")

    print("Assertions passed: sklearn and xorq predictions match.")

    # Generate deferred plots
    xo_plots = {}
    for kernel in KERNELS:
        plot_fn = _build_svm_kernel_plot(
            kernel=kernel,
            X_data=freeze(X.tolist()),
            y_data=freeze(y.tolist()),
            pipe_template=xo_results[kernel]["sklearn_pipe"],
        )
        xo_plots[kernel] = deferred_matplotlib_plot(
            xo_results[kernel]["preds"], plot_fn
        ).execute()

    # Build composite plot: 2 rows x 4 cols
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i, kernel in enumerate(KERNELS):
        # Top row: sklearn
        clf = sk_results[kernel]["clf"]
        _plot_decision_boundary(
            axes[0, i], clf, X, y, kernel, X_MIN, X_MAX, Y_MIN, Y_MAX
        )
        if i == 0:
            axes[0, i].set_ylabel("sklearn", fontsize=12, fontweight="bold")

        # Bottom row: xorq
        img = load_plot_bytes(xo_plots[kernel])
        axes[1, i].imshow(img)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("xorq", fontsize=12, fontweight="bold")

    fig.suptitle("SVM Kernels Comparison: sklearn vs xorq", fontsize=16, y=0.995)
    fig.tight_layout()
    out = "imgs/svm_kernels.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
