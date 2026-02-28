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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.tree import DecisionTreeRegressor

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    split_data_nop,
)
from xorq_gallery.utils import save_fig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
N_SAMPLES = 80
FEATURE_COLS = ("x",)
TARGET_COL = "y"
PRED_COL = "pred"
MODEL_COLORS = {2: "darkorange", 5: "cornflowerblue"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Generate synthetic 1D sine wave data with noise (80 training samples)."""
    np.random.seed(RANDOM_SEED)
    X = np.sort(5 * np.random.rand(N_SAMPLES, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - np.random.rand(16))
    return pd.DataFrame({"x": X[:, 0], TARGET_COL: y})


def _fine_grid():
    """Dense prediction grid for smooth fitted-curve visualization."""
    return pd.DataFrame({"x": np.arange(0.0, 5.0, 0.01)})


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_tree_result(ax, X_train, y_train, X_fine, y_pred, max_depth, color):
    ax.plot(X_fine, y_pred, color=color, label=f"max_depth={max_depth}", linewidth=2)
    ax.scatter(
        X_train, y_train, s=20, edgecolor="black", c="darkorange", label="training data"
    )
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title(f"Decision Tree Regression (max_depth={max_depth})")
    ax.legend()


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def compare_results(comparator):
    print("\n=== Comparing Results ===")
    for name, _ in comparator.names_pipelines:
        sk_tree = comparator.sklearn_results[name]["fitted"]
        xo_preds_df = comparator.xorq_results[name]["preds"]
        X = comparator.df[list(FEATURE_COLS)]
        sk_preds = sk_tree.predict(X)
        xo_preds = xo_preds_df[PRED_COL].values
        np.testing.assert_allclose(sk_preds, xo_preds, rtol=1e-10)
        print(f"  {name}: n_leaves={sk_tree.get_n_leaves()}, predictions match sklearn")


def plot_results(comparator):
    df = comparator.df
    X_train = df["x"].values
    y_train = df[TARGET_COL].values
    df_fine = _fine_grid()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for col_idx, (name, _) in enumerate(comparator.names_pipelines):
        sk_tree = comparator.sklearn_results[name]["fitted"]
        max_depth = sk_tree.get_depth()
        color = MODEL_COLORS[max_depth]

        # Top row: sklearn — predict on fine grid directly
        y_sk = sk_tree.predict(df_fine[list(FEATURE_COLS)])
        _plot_tree_result(
            axes[0, col_idx],
            X_train,
            y_train,
            df_fine["x"].values,
            y_sk,
            max_depth,
            color,
        )
        if col_idx == 0:
            axes[0, col_idx].text(
                -0.3,
                0.5,
                "sklearn",
                transform=axes[0, col_idx].transAxes,
                fontsize=12,
                fontweight="bold",
                va="center",
                rotation=90,
            )

        # Bottom row: xorq — predict on fine grid via xorq_fitted
        xorq_fitted = comparator.deferred_xorq_results[name]["xorq_fitted"]
        xo_preds_df = xorq_fitted.predict(xo.memtable(df_fine), name=PRED_COL).execute()
        y_xo = xo_preds_df[PRED_COL].values
        _plot_tree_result(
            axes[1, col_idx],
            X_train,
            y_train,
            df_fine["x"].values,
            y_xo,
            max_depth,
            color,
        )
        if col_idx == 0:
            axes[1, col_idx].text(
                -0.3,
                0.5,
                "xorq",
                transform=axes[1, col_idx].transAxes,
                fontsize=12,
                fontweight="bold",
                va="center",
                rotation=90,
            )

    fig.suptitle("Decision Tree Regression: sklearn vs xorq", fontsize=16, y=0.995)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = (DEPTH_2, DEPTH_5) = ("depth_2", "depth_5")
names_pipelines = (
    (
        DEPTH_2,
        SklearnPipeline(
            [("tree", DecisionTreeRegressor(max_depth=2, random_state=RANDOM_SEED))]
        ),
    ),
    (
        DEPTH_5,
        SklearnPipeline(
            [("tree", DecisionTreeRegressor(max_depth=5, random_state=RANDOM_SEED))]
        ),
    ),
)
metrics_names_funcs = ()

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
# expose the exprs to invoke `xorq build plot_tree_regression.py --expr $expr_name`
(xorq_depth2_preds, xorq_depth5_preds) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    comparator.result_comparison
    save_fig("imgs/tree_regression.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
