"""Decision Tree Regression
===========================

sklearn: Train DecisionTreeRegressor models with max_depth of 2 and 5 on
synthetic 1D noisy sine data. Fit each model, predict on test points, and
plot fitted curves to demonstrate overfitting with deeper trees.

xorq: Same DecisionTreeRegressor models wrapped in Pipeline.from_instance,
fit/predict deferred, generate deferred regression plots via
deferred_matplotlib_plot.

Both produce identical predictions.

Dataset: Synthetic 1D sine wave with noise (80 samples)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.tree import DecisionTreeRegressor
from toolz import curry
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
N_SAMPLES = 80
MAX_DEPTHS = (2, 5)
MODEL_COLORS = {2: "darkorange", 5: "cornflowerblue"}
MODEL_LABELS = {2: "max_depth=2", 5: "max_depth=5"}


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Generate synthetic 1D sine wave data with noise.

    Matches the sklearn decision tree regression example: 80 training samples
    sorted along x-axis, sine wave + noise.
    """
    np.random.seed(RANDOM_SEED)
    X = np.sort(5 * np.random.rand(N_SAMPLES, 1), axis=0)
    y = np.sin(X).ravel()

    # Add noise to targets
    y[::5] += 3 * (0.5 - np.random.rand(16))

    # Create test data
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

    df = pd.DataFrame(X, columns=["x"])
    df["y"] = y

    df_test = pd.DataFrame(X_test, columns=["x"])

    return df, df_test


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_tree_result(ax, X, y, X_test, y_pred, max_depth, color):
    """Plot decision tree fitted curve.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    X : numpy.ndarray
        Training feature array (N, 1).
    y : numpy.ndarray
        Training target array (N,).
    X_test : numpy.ndarray
        Test feature array (M, 1).
    y_pred : numpy.ndarray
        Predicted values on test data (M,).
    max_depth : int
        Max depth of the tree for labeling.
    color : str
        Color for the fitted line.
    """
    # Plot fitted curve
    ax.plot(
        X_test,
        y_pred,
        color=color,
        label=MODEL_LABELS[max_depth],
        linewidth=2,
    )

    # Plot training data
    ax.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="training data")

    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title(f"Decision Tree Regression (max_depth={max_depth})")
    ax.legend()


@curry
def _build_tree_plot_deferred(pred_df, max_depth, X, y, X_test):
    """Build tree regression plot from materialized predictions."""
    y_pred = pred_df["pred"].values
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_tree_result(ax, X, y, X_test, y_pred, max_depth, MODEL_COLORS[max_depth])
    fig.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit and predict
# =========================================================================


def sklearn_way(df, df_test):
    """Eager sklearn: fit DecisionTreeRegressor with each max_depth, predict.

    Returns dict of max_depth -> {tree, predictions}.
    """
    X = df[["x"]].values
    y = df["y"].values
    X_test = df_test[["x"]].values

    # max_depth=2
    tree_2 = DecisionTreeRegressor(max_depth=2, random_state=RANDOM_SEED)
    tree_2.fit(X, y)
    y_pred_2 = tree_2.predict(X_test)
    print(f"  sklearn: max_depth=2 | n_leaves={tree_2.get_n_leaves()}")

    # max_depth=5
    tree_5 = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_SEED)
    tree_5.fit(X, y)
    y_pred_5 = tree_5.predict(X_test)
    print(f"  sklearn: max_depth=5 | n_leaves={tree_5.get_n_leaves()}")

    return {
        2: {"tree": tree_2, "predictions": y_pred_2, "n_leaves": tree_2.get_n_leaves()},
        5: {"tree": tree_5, "predictions": y_pred_5, "n_leaves": tree_5.get_n_leaves()},
    }


# =========================================================================
# XORQ WAY -- deferred fit and predict
# =========================================================================


def xorq_way(df, df_test):
    """Deferred xorq: wrap DecisionTreeRegressor in sklearn Pipeline, then
    Pipeline.from_instance, fit deferred, return predictions.

    Returns dict of max_depth -> predictions_expr.
    """
    con = xo.connect()
    table = con.register(df, "tree_data")
    table_test = con.register(df_test, "tree_test_data")

    # max_depth=2
    sklearn_pipe_2 = SklearnPipeline([
        ("tree", DecisionTreeRegressor(max_depth=2, random_state=RANDOM_SEED))
    ])
    xorq_pipe_2 = Pipeline.from_instance(sklearn_pipe_2)
    fitted_2 = xorq_pipe_2.fit(table, features=("x",), target="y")
    preds_2 = fitted_2.predict(table_test, name="pred")

    # max_depth=5
    sklearn_pipe_5 = SklearnPipeline([
        ("tree", DecisionTreeRegressor(max_depth=5, random_state=RANDOM_SEED))
    ])
    xorq_pipe_5 = Pipeline.from_instance(sklearn_pipe_5)
    fitted_5 = xorq_pipe_5.fit(table, features=("x",), target="y")
    preds_5 = fitted_5.predict(table_test, name="pred")

    return {2: preds_2, 5: preds_5}


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df, df_test = _load_data()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(df, df_test)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(df, df_test)

    # Execute deferred predictions and verify they match
    print("\n=== ASSERTIONS ===")
    xo_preds_df_2 = xo_results[2].execute()
    sk_preds_2 = sk_results[2]["predictions"]
    xo_preds_2 = xo_preds_df_2["pred"].values
    np.testing.assert_allclose(sk_preds_2, xo_preds_2, rtol=1e-10)
    print(f"  xorq:   max_depth=2 | predictions match sklearn")

    xo_preds_df_5 = xo_results[5].execute()
    sk_preds_5 = sk_results[5]["predictions"]
    xo_preds_5 = xo_preds_df_5["pred"].values
    np.testing.assert_allclose(sk_preds_5, xo_preds_5, rtol=1e-10)
    print(f"  xorq:   max_depth=5 | predictions match sklearn")

    print("Assertions passed: sklearn and xorq predictions match.")

    # Build composite plot: 2 rows x 2 cols (one row per approach)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    X = df[["x"]].values
    y = df["y"].values
    X_test = df_test[["x"]].values

    # Top row: sklearn results
    ax_sk_2 = axes[0, 0]
    y_pred_sk_2 = sk_results[2]["predictions"]
    _plot_tree_result(ax_sk_2, X, y, X_test, y_pred_sk_2, 2, MODEL_COLORS[2])
    ax_sk_2.text(
        -0.3, 0.5, "sklearn", transform=ax_sk_2.transAxes,
        fontsize=12, fontweight="bold", va="center", rotation=90,
    )

    ax_sk_5 = axes[0, 1]
    y_pred_sk_5 = sk_results[5]["predictions"]
    _plot_tree_result(ax_sk_5, X, y, X_test, y_pred_sk_5, 5, MODEL_COLORS[5])

    # Bottom row: xorq results (execute deferred plots)
    png_bytes_2 = deferred_matplotlib_plot(
        xo_results[2], _build_tree_plot_deferred(max_depth=2, X=X, y=y, X_test=X_test)
    ).execute()
    img_2 = load_plot_bytes(png_bytes_2)
    ax_xo_2 = axes[1, 0]
    ax_xo_2.imshow(img_2)
    ax_xo_2.axis("off")
    ax_xo_2.text(
        -0.3, 0.5, "xorq", transform=ax_xo_2.transAxes,
        fontsize=12, fontweight="bold", va="center", rotation=90,
    )

    png_bytes_5 = deferred_matplotlib_plot(
        xo_results[5], _build_tree_plot_deferred(max_depth=5, X=X, y=y, X_test=X_test)
    ).execute()
    img_5 = load_plot_bytes(png_bytes_5)
    ax_xo_5 = axes[1, 1]
    ax_xo_5.imshow(img_5)
    ax_xo_5.axis("off")

    fig.suptitle(
        "Decision Tree Regression: sklearn vs xorq", fontsize=16, y=0.995
    )
    fig.tight_layout()
    out = "imgs/tree_regression.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
