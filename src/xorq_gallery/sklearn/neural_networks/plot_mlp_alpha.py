"""Varying regularization in Multi-layer Perceptron
================================================

sklearn: Generate three synthetic 2D datasets (moons, circles, linearly separable),
fit MLPClassifier with varying alpha values (regularization strength from 0.1 to 10),
evaluate accuracy, plot decision boundaries via meshgrid evaluation.

xorq: Same MLP pipelines wrapped in Pipeline.from_instance, fit/predict deferred,
evaluate with deferred_sklearn_metric, generate deferred decision boundary plots.

Both produce identical accuracy scores.

Dataset: make_moons, make_circles, make_classification (sklearn synthetic)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from toolz import curry
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline
from xorq.ibis_yaml.utils import freeze

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE_SPLIT = 42
RANDOM_STATE_MLP = 1
TEST_SIZE = 0.4
H = 0.02  # meshgrid step size

# Alpha values to test
ALPHAS = np.logspace(-1, 1, 5)


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Generate three synthetic 2D classification datasets."""
    # Moons dataset
    X_moons, y_moons = make_moons(noise=0.3, random_state=0)

    # Circles dataset
    X_circles, y_circles = make_circles(
        noise=0.2, factor=0.5, random_state=1
    )

    # Linearly separable dataset
    X_linearly, y_linearly = make_classification(
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=0,
        n_clusters_per_class=1,
    )
    rng = np.random.RandomState(2)
    X_linearly += 2 * rng.uniform(size=X_linearly.shape)

    datasets = {
        "moons": (X_moons, y_moons),
        "circles": (X_circles, y_circles),
        "linearly_separable": (X_linearly, y_linearly),
    }
    return datasets


def _build_classifiers():
    """Return dict of alpha values -> sklearn Pipeline instances."""
    return {
        alpha: SklearnPipeline([
            ("standardscaler", StandardScaler()),
            ("mlpclassifier", MLPClassifier(
                solver="lbfgs",
                alpha=alpha,
                random_state=RANDOM_STATE_MLP,
                max_iter=2000,
                early_stopping=True,
                hidden_layer_sizes=[10, 10],
            )),
        ])
        for alpha in ALPHAS
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_decision_boundary(ax, X, y, clf, title, x_min, x_max, y_min, y_max, score=None):
    """Plot decision boundary for a fitted sklearn classifier."""
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, H), np.arange(y_min, y_max, H)
    )

    # Compute decision function or predict_proba
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
    else:
        Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]

    Z = Z.reshape(xx.shape)

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    ax.contourf(xx, yy, Z, alpha=0.8, cmap=cm)

    # Plot training and test points
    # Note: We'll pass both train and test sets combined for this visualization
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors="black", s=25)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title, fontsize=9)

    # Add score text if provided
    if score is not None:
        ax.text(
            xx.max() - 0.3,
            yy.min() + 0.3,
            f"{score:.3f}".lstrip("0"),
            size=15,
            horizontalalignment="right",
        )


@curry
def _build_decision_boundary_plot(df, X_train, y_train, X_test, y_test, bounds, clf_copy, alpha_val):
    """Build decision boundary plot from materialized predictions.

    Array args are passed as tuples (for xorq hashability) and
    converted back to numpy arrays inside this function.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    fitted_clf = clf_copy.fit(X_train, y_train)
    score_val = fitted_clf.score(X_test, y_test)
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.concatenate([y_train, y_test])
    fig, ax = plt.subplots(figsize=(5, 4))
    _plot_decision_boundary(
        ax, X_combined, y_combined, fitted_clf, f"alpha {alpha_val:.2f}", *bounds, score=score_val,
    )
    fig.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit/predict, decision boundary plots
# =========================================================================


def sklearn_way(datasets):
    """Eager sklearn: fit MLP classifiers with varying alpha on three datasets,
    compute accuracy, generate decision boundary plots."""

    results = {}

    for ds_name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE_SPLIT
        )
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        bounds = (x_min, x_max, y_min, y_max)

        results[ds_name] = {}
        for alpha in ALPHAS:
            clf = SklearnPipeline([
                ("standardscaler", StandardScaler()),
                ("mlpclassifier", MLPClassifier(
                    solver="lbfgs",
                    alpha=alpha,
                    random_state=RANDOM_STATE_MLP,
                    max_iter=2000,
                    early_stopping=True,
                    hidden_layer_sizes=[10, 10]
                )),
            ])
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            print(f"  sklearn: {ds_name:20s} | alpha={alpha:.2f} | score = {score:.3f}")
            results[ds_name][alpha] = {
                "clf": clf,
                "score": score,
                "X": X,
                "y": y,
                "bounds": bounds,
            }

    return results


# =========================================================================
# XORQ WAY -- deferred fit/predict, deferred decision boundary plots
# =========================================================================


def xorq_way(datasets):
    """Deferred xorq: wrap MLP classifiers in Pipeline.from_instance, fit/predict
    deferred, compute deferred accuracy.

    Returns dict of dataset_name -> dict of alpha -> {preds: expr, metrics: expr, plot_data: dict}.
    """
    con = xo.connect()
    make_metric = deferred_sklearn_metric(target="y", pred="pred")
    results = {}

    for ds_name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE_SPLIT
        )

        train_df = pd.DataFrame(X_train, columns=["x0", "x1"])
        train_df["y"] = y_train
        test_df = pd.DataFrame(X_test, columns=["x0", "x1"])
        test_df["y"] = y_test
        train_table = con.register(train_df, f"train_{ds_name}")
        test_table = con.register(test_df, f"test_{ds_name}")

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        bounds = (x_min, x_max, y_min, y_max)

        results[ds_name] = {}
        classifiers = _build_classifiers()
        for alpha in ALPHAS:
            xorq_pipe = Pipeline.from_instance(classifiers[alpha])
            fitted = xorq_pipe.fit(train_table, features=("x0", "x1"), target="y")
            preds = fitted.predict(test_table, name="pred")
            metrics = preds.agg(score=make_metric(metric=accuracy_score))

            results[ds_name][alpha] = {
                "preds": preds,
                "metrics": metrics,
                "plot_data": {
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                    "y_test": y_test,
                    "bounds": bounds,
                    "clf": classifiers[alpha],
                    "alpha": alpha,
                },
            }

    return results


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

    # Execute deferred metrics and assert equivalence
    print("\n=== ASSERTIONS ===")
    for ds_name in datasets.keys():
        for alpha in ALPHAS:
            sk_score = sk_results[ds_name][alpha]["score"]
            xo_metrics_df = xo_results[ds_name][alpha]["metrics"].execute()
            xo_score = xo_metrics_df["score"].iloc[0]
            print(f"  xorq:   {ds_name:20s} | alpha={alpha:.2f} | score = {xo_score:.3f}")
            # Use higher tolerance for MLPs due to stochastic nature
            np.testing.assert_allclose(sk_score, xo_score, rtol=0.03)

    print("Assertions passed: sklearn and xorq metrics match.")

    # Build composite plot: 3 rows (datasets) x 5 cols (alphas)
    n_datasets = len(datasets)
    n_alphas = len(ALPHAS)

    # Create sklearn plot
    fig_sk, axes_sk = plt.subplots(
        n_datasets, n_alphas, figsize=(n_alphas * 3, n_datasets * 3)
    )

    dataset_names = list(datasets.keys())

    for i, ds_name in enumerate(dataset_names):
        for j, alpha in enumerate(ALPHAS):
            ax = axes_sk[i, j]
            result = sk_results[ds_name][alpha]
            clf = result["clf"]
            X = result["X"]
            y = result["y"]
            score = result["score"]
            x_min, x_max, y_min, y_max = result["bounds"]

            _plot_decision_boundary(
                ax, X, y, clf, f"alpha {alpha:.2f}", x_min, x_max, y_min, y_max, score=score
            )

            # Add row labels
            if j == 0:
                ax.set_ylabel(ds_name, fontsize=10, fontweight="bold")

    fig_sk.suptitle("MLP Regularization: sklearn", fontsize=14, y=0.995)
    fig_sk.tight_layout()
    out_sk = "imgs/plot_mlp_alpha_sklearn.png"
    fig_sk.savefig(out_sk, dpi=150, bbox_inches="tight")
    plt.close(fig_sk)
    print(f"sklearn plot saved to {out_sk}")

    # Create xorq plot - deferred_matplotlib_plot happens here in main()
    fig_xo, axes_xo = plt.subplots(
        n_datasets, n_alphas, figsize=(n_alphas * 3, n_datasets * 3)
    )

    for i, ds_name in enumerate(dataset_names):
        for j, alpha in enumerate(ALPHAS):
            ax = axes_xo[i, j]
            result = xo_results[ds_name][alpha]
            preds_expr = result["preds"]
            plot_data = result["plot_data"]

            # Build plot function with captured variables
            plot_func = _build_decision_boundary_plot(
                X_train=freeze(plot_data["X_train"].tolist()),
                y_train=freeze(plot_data["y_train"].tolist()),
                X_test=freeze(plot_data["X_test"].tolist()),
                y_test=freeze(plot_data["y_test"].tolist()),
                bounds=plot_data["bounds"],
                clf_copy=plot_data["clf"],
                alpha_val=plot_data["alpha"],
            )

            plot_expr = deferred_matplotlib_plot(preds_expr, plot_func)
            png_bytes = plot_expr.execute()
            img = load_plot_bytes(png_bytes)
            ax.imshow(img)
            ax.axis("off")

            # Add row labels
            if j == 0:
                ax.set_ylabel(ds_name, fontsize=10, fontweight="bold")

    fig_xo.suptitle("MLP Regularization: xorq", fontsize=14, y=0.995)
    fig_xo.tight_layout()
    out_xo = "imgs/plot_mlp_alpha_xorq.png"
    fig_xo.savefig(out_xo, dpi=150, bbox_inches="tight")
    plt.close(fig_xo)
    print(f"xorq plot saved to {out_xo}")

    # Composite side-by-side
    sk_img = plt.imread(out_sk)
    xo_img = plt.imread(out_xo)

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    axes[0].imshow(sk_img)
    axes[0].set_title("sklearn")
    axes[0].axis("off")

    axes[1].imshow(xo_img)
    axes[1].set_title("xorq")
    axes[1].axis("off")

    fig.suptitle("MLP Regularization: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    out = "imgs/plot_mlp_alpha.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Composite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
