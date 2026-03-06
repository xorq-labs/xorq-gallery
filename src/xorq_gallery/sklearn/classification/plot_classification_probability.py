"""Plot Classification Probability
===================================

sklearn: Train several probabilistic classifiers (Logistic Regression with
varying regularization, Gradient Boosting) on the first two features of the
Iris dataset. Compute per-class predicted probabilities on a test set,
evaluate accuracy, and plot decision-boundary probability maps for each
class plus an overall "max class" column.

xorq: Same classifiers wrapped in Pipeline.from_instance, deferred
fit/predict, accuracy via deferred_sklearn_metric. Decision-boundary plots
refit classifiers inside deferred_matplotlib_plot.

Both produce equivalent accuracy metrics.

Dataset: Iris (sklearn) -- first two features only

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/classification/plot_classification_probability.py
"""

from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.datasets import load_iris
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from toolz import curry

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
)
from xorq_gallery.sklearn.sklearn_lib import (
    make_sklearn_result as _make_sklearn_result,
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

RANDOM_STATE = 42
TEST_SIZE = 0.5
FEATURE_COLS = ("sepal_length", "sepal_width")
TARGET_COL = "target"
PRED_COL = "pred"
CLASS_NAMES = ("setosa", "versicolor", "virginica")
H = 0.02  # meshgrid step size


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Load Iris dataset (first two features only) as DataFrame."""
    iris = load_iris()
    X = iris.data[:, :2]
    return pd.DataFrame(X, columns=list(FEATURE_COLS)).assign(
        **{TARGET_COL: iris.target}
    )


# ---------------------------------------------------------------------------
# Pipeline builders (shared between sklearn and xorq-refit)
# ---------------------------------------------------------------------------


def _build_classifiers():
    """Return list of (name, SklearnPipeline) matching names_pipelines order."""
    return [
        (
            LR_C01_NAME,
            SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(C=0.1, max_iter=1000)),
                ]
            ),
        ),
        (
            LR_C100_NAME,
            SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(C=100, max_iter=1000)),
                ]
            ),
        ),
        (
            HGB_NAME,
            SklearnPipeline(
                [
                    ("hgb", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
                ]
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_proba_grid(classifiers_results, class_names, title_prefix=""):
    """Plot per-class probability maps and a max-class column."""
    n_clf = len(classifiers_results)
    n_classes = len(class_names)
    n_cols = n_classes + 1

    fig, axes = plt.subplots(
        nrows=n_clf, ncols=n_cols, figsize=(n_cols * 2.6, n_clf * 2.6)
    )
    if n_clf == 1:
        axes = axes[np.newaxis, :]

    first = classifiers_results[0]
    X_all = first["X_train"]
    x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
    y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, H), np.arange(y_min, y_max, H))

    scatter_kw = dict(s=20, marker="o", linewidths=0.6, edgecolor="k", alpha=0.7)

    for row, res in enumerate(classifiers_results):
        clf = res["clf"]
        X_test = res["X_test"]
        y_test = res["y_test"]
        y_pred = res["y_pred"]
        grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
        probas = clf.predict_proba(grid_pts)

        for label in range(n_classes):
            ax = axes[row, label]
            Z = probas[:, label].reshape(xx.shape)
            ax.contourf(xx, yy, Z, levels=100, vmin=0, vmax=1, cmap="Blues")
            mask = y_pred == label
            ax.scatter(X_test[mask, 0], X_test[mask, 1], c="w", **scatter_kw)
            ax.set(xticks=(), yticks=())
            if row == 0:
                ax.set_title(f"Class {class_names[label]}", fontsize=9)

        ax_max = axes[row, n_classes]
        Z_max = probas.max(axis=1).reshape(xx.shape)
        ax_max.contourf(xx, yy, Z_max, levels=100, vmin=0, vmax=1, cmap="Blues")
        for label in range(n_classes):
            mask = y_test == label
            ax_max.scatter(
                X_test[mask, 0],
                X_test[mask, 1],
                **scatter_kw,
                c=[plt.get_cmap("tab10")(label)] * int(mask.sum()),
            )
        ax_max.set(xticks=(), yticks=())
        if row == 0:
            ax_max.set_title("Max class", fontsize=9)

        axes[row, 0].set_ylabel(res["name"], fontsize=8, fontweight="bold")

    fig.suptitle(f"{title_prefix}Classification Probability", fontsize=12)
    fig.tight_layout()
    return fig


@curry
def _build_proba_plot(df, class_names, feature_cols):
    """Refit classifiers on materialised DataFrame and build probability grid plot."""
    X = df[list(feature_cols)].values
    y = df[TARGET_COL].values
    results = []
    for name, clf in _build_classifiers():
        clf.fit(X, y)
        y_pred = clf.predict(X)
        results.append(
            {
                "name": name,
                "clf": clf,
                "X_train": X,
                "X_test": X,
                "y_test": y,
                "y_pred": y_pred,
            }
        )
    return _plot_proba_grid(results, class_names, title_prefix="xorq: ")


# ---------------------------------------------------------------------------
# make_other override: store full fitted pipeline for predict_proba on meshgrid
# ---------------------------------------------------------------------------


def _make_sklearn_other(fitted):
    return {"full_pipeline": fitted}


make_sklearn_result = _make_sklearn_result(make_other=_make_sklearn_other)


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
        print(f"  {name:35s} accuracy - sklearn: {sk_acc:.4f}, xorq: {xo_acc:.4f}")


def plot_results(comparator):
    train_df, _ = comparator.get_split_data()
    X_train = train_df[list(FEATURE_COLS)].values
    y_train = train_df[TARGET_COL].values

    sk_classifiers_results = [
        {
            "name": name,
            "clf": comparator.sklearn_results[name]["other"]["full_pipeline"],
            "X_train": X_train,
            "X_test": X_train,
            "y_test": y_train,
            "y_pred": comparator.sklearn_results[name]["other"][
                "full_pipeline"
            ].predict(X_train),
        }
        for name in methods
    ]
    sk_fig = _plot_proba_grid(sk_classifiers_results, CLASS_NAMES, "sklearn: ")

    xo_plot_fn = _build_proba_plot(
        class_names=CLASS_NAMES,
        feature_cols=FEATURE_COLS,
    )
    xo_png = deferred_matplotlib_plot(xo.memtable(train_df), xo_plot_fn).execute()

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(load_plot_bytes(xo_png))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    fig.suptitle("Classification Probability: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = (LR_C01_NAME, LR_C100_NAME, HGB_NAME) = (
    "Logistic Regression (C=0.1)",
    "Logistic Regression (C=100)",
    "Gradient Boosting",
)
names_pipelines = tuple((name, pipe) for name, pipe in _build_classifiers())
metrics_names_funcs = (("accuracy", accuracy_score),)

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=TARGET_COL,
    pred=PRED_COL,
    metrics_names_funcs=metrics_names_funcs,
    load_data=load_data,
    split_data=partial(
        train_test_split, test_size=TEST_SIZE, random_state=RANDOM_STATE
    ),
    make_sklearn_result=make_sklearn_result,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)
# expose the exprs to invoke `xorq build plot_classification_probability.py --expr $expr_name`
(xorq_lr_c01_preds, xorq_lr_c100_preds, xorq_hgb_preds) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    comparator.result_comparison
    save_fig("imgs/plot_classification_probability.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
