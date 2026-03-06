"""Varying regularization in Multi-layer Perceptron
================================================

sklearn: Generate three synthetic 2D datasets (moons, circles, linearly separable),
fit MLPClassifier with varying alpha values (regularization strength from 0.1 to 10),
evaluate accuracy, plot decision boundaries via meshgrid evaluation.

xorq: Same MLP pipelines wrapped in Pipeline.from_instance, fit/predict deferred,
evaluate with deferred_sklearn_metric, generate deferred decision boundary plots.

Both produce identical accuracy scores.

Dataset: make_moons, make_circles, make_classification (sklearn synthetic)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/neural_networks/plot_mlp_alpha.py
"""

from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

from xorq_gallery.sklearn.sklearn_lib import SklearnXorqComparator
from xorq_gallery.utils import fig_to_image, save_fig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE_SPLIT = 42
RANDOM_STATE_MLP = 1
TEST_SIZE = 0.4
H = 0.02  # meshgrid step size

FEATURE_COLS = ("x0", "x1")
TARGET_COL = "y"
PRED_COL = "pred"

# Alpha values to test
ALPHAS = np.logspace(-1, 1, 5)


# ---------------------------------------------------------------------------
# Data loading — one loader per dataset
# ---------------------------------------------------------------------------


def load_data_moons():
    X, y = make_moons(noise=0.3, random_state=0)
    df = pd.DataFrame(X, columns=list(FEATURE_COLS))
    df[TARGET_COL] = y
    return df


def load_data_circles():
    X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
    df = pd.DataFrame(X, columns=list(FEATURE_COLS))
    df[TARGET_COL] = y
    return df


def load_data_linearly_separable():
    X, y = make_classification(
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=0,
        n_clusters_per_class=1,
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    df = pd.DataFrame(X, columns=list(FEATURE_COLS))
    df[TARGET_COL] = y
    return df


_LOAD_DATA_FNS = {
    "moons": load_data_moons,
    "circles": load_data_circles,
    "linearly_separable": load_data_linearly_separable,
}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_decision_boundary(
    ax, X, y, clf, title, x_min, x_max, y_min, y_max, score=None
):
    """Plot decision boundary for a fitted sklearn classifier."""
    xx, yy = np.meshgrid(np.arange(x_min, x_max, H), np.arange(y_min, y_max, H))

    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
    else:
        Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]

    Z = Z.reshape(xx.shape)

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    ax.contourf(xx, yy, Z, alpha=0.8, cmap=cm)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors="black", s=25)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title, fontsize=9)

    if score is not None:
        ax.text(
            xx.max() - 0.3,
            yy.min() + 0.3,
            f"{score:.3f}".lstrip("0"),
            size=15,
            horizontalalignment="right",
        )


def _draw_boundary(ax, comparator, name, title, show_score=True):
    """Draw decision boundary for a single fitted estimator into *ax*."""
    fitted = comparator.sklearn_results[name]["fitted"]
    df = comparator.df
    X = df[list(FEATURE_COLS)].values
    y = df[TARGET_COL].values
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    score = (
        comparator.sklearn_results[name]["metrics"]["accuracy"] if show_score else None
    )
    _plot_decision_boundary(
        ax, X, y, fitted, title, x_min, x_max, y_min, y_max, score=score
    )


def _draw_boundary_xorq(ax, comparator, name, title, show_score=True):
    """Draw decision boundary for the xorq side into *ax*.

    The xorq fitted model is the same sklearn estimator under the hood,
    so we use it to draw the meshgrid decision boundary directly.
    """
    fitted = comparator.xorq_results[name]["fitted"]
    df = comparator.df
    X = df[list(FEATURE_COLS)].values
    y = df[TARGET_COL].values
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    score = comparator.xorq_results[name]["metrics"]["accuracy"] if show_score else None
    _plot_decision_boundary(
        ax, X, y, fitted, title, x_min, x_max, y_min, y_max, score=score
    )


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def compare_results(comparator):
    print("\n=== Comparing Results ===")
    for name in comparator.sklearn_results:
        sk_acc = comparator.sklearn_results[name]["metrics"]["accuracy"]
        xo_acc = comparator.xorq_results[name]["metrics"]["accuracy"]
        print(f"  {name:10s} accuracy — sklearn: {sk_acc:.4f}, xorq: {xo_acc:.4f}")
        np.testing.assert_allclose(sk_acc, xo_acc, rtol=0.03)


def plot_results(comparator):
    """Build a row of 5 sklearn + 5 xorq decision boundary subplots."""
    n_alphas = len(ALPHAS)
    n_cols = n_alphas * 2
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.6, 2.8))

    for j, name in enumerate(alpha_names):
        _draw_boundary(axes[j], comparator, name, f"alpha {ALPHAS[j]:.2f}")
        _draw_boundary_xorq(
            axes[n_alphas + j], comparator, name, f"alpha {ALPHAS[j]:.2f}"
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup — one comparator per dataset
# ---------------------------------------------------------------------------

dataset_names = ("moons", "circles", "linearly_separable")
alpha_names = tuple(f"alpha_{alpha:.2f}" for alpha in ALPHAS)

names_pipelines = tuple(
    (
        name,
        SklearnPipeline(
            [
                ("standardscaler", StandardScaler()),
                (
                    "mlpclassifier",
                    MLPClassifier(
                        solver="lbfgs",
                        alpha=alpha,
                        random_state=RANDOM_STATE_MLP,
                        max_iter=2000,
                        early_stopping=True,
                        hidden_layer_sizes=[10, 10],
                    ),
                ),
            ]
        ),
    )
    for name, alpha in zip(alpha_names, ALPHAS)
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
        split_data=partial(
            train_test_split, test_size=TEST_SIZE, random_state=RANDOM_STATE_SPLIT
        ),
        compare_results_fn=compare_results,
        plot_results_fn=plot_results,
    )
    for ds_name in dataset_names
}

# expose the exprs to invoke `xorq build plot_mlp_alpha.py --expr $expr_name`
(
    xorq_moons_alpha010_preds,
    xorq_moons_alpha032_preds,
    xorq_moons_alpha100_preds,
    xorq_moons_alpha316_preds,
    xorq_moons_alpha1000_preds,
) = (comparators["moons"].deferred_xorq_results[name]["preds"] for name in alpha_names)
(
    xorq_circles_alpha010_preds,
    xorq_circles_alpha032_preds,
    xorq_circles_alpha100_preds,
    xorq_circles_alpha316_preds,
    xorq_circles_alpha1000_preds,
) = (
    comparators["circles"].deferred_xorq_results[name]["preds"] for name in alpha_names
)
(
    xorq_linearly_alpha010_preds,
    xorq_linearly_alpha032_preds,
    xorq_linearly_alpha100_preds,
    xorq_linearly_alpha316_preds,
    xorq_linearly_alpha1000_preds,
) = (
    comparators["linearly_separable"].deferred_xorq_results[name]["preds"]
    for name in alpha_names
)


# =========================================================================
# Main
# =========================================================================


def main():
    for ds_name in dataset_names:
        comparators[ds_name].result_comparison

    row_figs = [comparators[ds_name].plot_results() for ds_name in dataset_names]

    n_alphas = len(ALPHAS)
    n_cols = n_alphas * 2
    fig, axes = plt.subplots(
        len(dataset_names), 1, figsize=(n_cols * 2.6, len(dataset_names) * 3)
    )
    for row, (row_fig, ds_name) in enumerate(zip(row_figs, dataset_names)):
        axes[row].imshow(fig_to_image(row_fig))
        axes[row].axis("off")
        plt.close(row_fig)

    fig.subplots_adjust(top=0.88, left=0.06, hspace=0.08)
    fig.suptitle("MLP Regularization: sklearn vs xorq", fontsize=16, y=0.98)

    # Group labels
    fig.text(0.27, 0.92, "sklearn", ha="center", fontsize=13, fontweight="bold")
    fig.text(0.75, 0.92, "xorq", ha="center", fontsize=13, fontweight="bold")

    # Row labels (dataset names)
    for row, ds_name in enumerate(dataset_names):
        row_center = (
            axes[row].get_position().y0
            + (axes[row].get_position().y1 - axes[row].get_position().y0) / 2
        )
        fig.text(
            0.02,
            row_center,
            ds_name,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            rotation=90,
        )

    save_fig("imgs/plot_mlp_alpha.png", fig)


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
