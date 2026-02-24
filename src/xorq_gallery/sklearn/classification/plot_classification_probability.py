"""Plot Classification Probability
===================================

sklearn: Train several probabilistic classifiers (Logistic Regression with
varying regularization, Gradient Boosting) on the first two features of the
Iris dataset.  Compute per-class predicted probabilities on a test set,
evaluate accuracy / ROC-AUC / log-loss, and plot decision-boundary probability
maps for each class plus an overall "max class" column.

xorq: Same classifiers wrapped in Pipeline.from_instance, deferred
fit / predict_proba, deferred accuracy via deferred_sklearn_metric, and
deferred decision-boundary plots via deferred_matplotlib_plot.

Both produce equivalent evaluation metrics.

Dataset: Iris (sklearn) -- first two features only
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.datasets import load_iris
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from toolz import curry
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.5
TARGET_COL = "target"
PRED_COL = "pred"
PROBA_COL = "predict_proba"
ROW_IDX = "row_idx"
H = 0.02  # meshgrid step size


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Load Iris dataset (first two features only) and return as DataFrame."""
    iris = load_iris()
    X = iris.data[:, :2]  # first two features for 2-D visualisation
    y = iris.target

    feature_cols = ("sepal_length", "sepal_width")
    df = pd.DataFrame(X, columns=feature_cols)
    df[TARGET_COL] = y
    df[ROW_IDX] = range(len(df))

    return df, feature_cols, iris.target_names


def _build_classifiers():
    """Return list of (SklearnPipeline, display_name) tuples.

    Uses explicit Pipeline([("step", ...)]) -- never make_pipeline.
    """
    return [
        (
            SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(C=0.1, max_iter=1000)),
                ]
            ),
            "Logistic Regression (C=0.1)",
        ),
        (
            SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(C=100, max_iter=1000)),
                ]
            ),
            "Logistic Regression (C=100)",
        ),
        (
            SklearnPipeline(
                [
                    (
                        "hgb",
                        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
                    ),
                ]
            ),
            "Gradient Boosting",
        ),
    ]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_proba_grid(classifiers_results, class_names, title_prefix=""):
    """Plot per-class probability maps and a max-class column.

    classifiers_results: list of dicts with keys
        name, clf (fitted), X_train, X_test, y_test, y_pred
    """
    n_clf = len(classifiers_results)
    n_classes = len(class_names)
    n_cols = n_classes + 1  # one per class + max-class column

    fig, axes = plt.subplots(
        nrows=n_clf, ncols=n_cols, figsize=(n_cols * 2.6, n_clf * 2.6)
    )
    if n_clf == 1:
        axes = axes[np.newaxis, :]

    # Shared meshgrid bounds across all classifiers
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

        # Per-class probability maps
        for label in range(n_classes):
            ax = axes[row, label]
            Z = probas[:, label].reshape(xx.shape)
            ax.contourf(xx, yy, Z, levels=100, vmin=0, vmax=1, cmap="Blues")
            # Show test points predicted as this class
            mask = y_pred == label
            ax.scatter(X_test[mask, 0], X_test[mask, 1], c="w", **scatter_kw)
            ax.set(xticks=(), yticks=())
            if row == 0:
                ax.set_title(f"Class {class_names[label]}", fontsize=9)

        # Max-class column
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

        # Row label
        axes[row, 0].set_ylabel(res["name"], fontsize=8, fontweight="bold")

    fig.suptitle(f"{title_prefix}Classification Probability", fontsize=12)
    fig.tight_layout()
    return fig


@curry
def _build_proba_plot(df, class_names, feature_cols):
    """Curried plot function for deferred_matplotlib_plot.

    Rebuilds classifiers internally (avoids unhashable sklearn objects in
    curry kwargs), fits on the materialised DataFrame, and produces the
    probability grid plot.  Returns the Figure.
    """
    X = df[list(feature_cols)].values
    y = df[TARGET_COL].values

    results = []
    for clf, name in _build_classifiers():
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

    fig = _plot_proba_grid(results, class_names, title_prefix="xorq: ")
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit/predict_proba, evaluation metrics
# =========================================================================


def sklearn_way(train_df, test_df, feature_cols, class_names):
    """Eager sklearn: fit classifiers, predict, evaluate metrics.

    Returns dict with fitted-classifier results and evaluation metrics.
    No plotting here -- that belongs in main().
    """
    X_train = train_df[list(feature_cols)].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[list(feature_cols)].values
    y_test = test_df[TARGET_COL].values

    clf_list = _build_classifiers()
    classifiers_results = []
    eval_results = []

    for clf, name in clf_list:
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_proba_test = clf.predict_proba(X_test)
        y_pred_train = clf.predict(X_train)

        acc = accuracy_score(y_test, y_pred_test)
        auc_val = roc_auc_score(y_test, y_proba_test, multi_class="ovr")
        ll = log_loss(y_test, y_proba_test)

        print(
            f"  sklearn: {name:35s} | acc={acc:.3f}  auc={auc_val:.3f}  log_loss={ll:.3f}"
        )
        eval_results.append(
            {"name": name, "accuracy": acc, "roc_auc": auc_val, "log_loss": ll}
        )
        # Plot uses train data for both grid bounds and scatter overlay
        # so that the visual matches the xorq deferred plot (which only
        # receives train_data).
        classifiers_results.append(
            {
                "name": name,
                "clf": clf,
                "X_train": X_train,
                "X_test": X_train,
                "y_test": y_train,
                "y_pred": y_pred_train,
            }
        )

    return {"eval": eval_results, "classifiers_results": classifiers_results}


# =========================================================================
# XORQ WAY -- deferred fit/predict_proba, deferred metrics
# =========================================================================


def xorq_way(train_data, test_data, feature_cols):
    """Deferred xorq: Pipeline.from_instance + predict_proba + deferred metrics.

    Returns dict of classifier_name -> {preds, proba_expr, metrics}.
    No .execute() calls here.
    """
    make_accuracy = deferred_sklearn_metric(target=TARGET_COL, pred=PRED_COL)

    clf_list = _build_classifiers()
    results = {}

    for clf, name in clf_list:
        xorq_pipe = Pipeline.from_instance(clf)
        fitted = xorq_pipe.fit(
            train_data, features=tuple(feature_cols), target=TARGET_COL
        )
        preds = fitted.predict(test_data, name=PRED_COL)
        proba_expr = fitted.predict_proba(test_data, name=PROBA_COL)
        metrics = preds.agg(accuracy=make_accuracy(metric=accuracy_score))

        results[name] = {
            "preds": preds,
            "proba_expr": proba_expr,
            "metrics": metrics,
        }

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df, feature_cols, class_names = _load_data()

    # Hash-based split via xorq -- single source of truth
    con = xo.connect()
    table = con.register(df, "iris_2d")
    train_data, test_data = xo.train_test_splits(
        table,
        test_sizes=TEST_SIZE,
        unique_key=ROW_IDX,
        random_seed=RANDOM_STATE,
    )
    train_data = train_data.order_by(ROW_IDX)
    test_data = test_data.order_by(ROW_IDX)

    # Materialise for sklearn
    train_df = train_data.execute()
    test_df = test_data.execute()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(train_df, test_df, feature_cols, class_names)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(train_data, test_data, feature_cols)

    # --- Execute deferred metrics and assert equivalence ---
    print("\n=== ASSERTIONS ===")
    sk_eval = {r["name"]: r for r in sk_results["eval"]}

    for name, res in xo_results.items():
        xo_metrics = res["metrics"].execute()
        xo_acc = xo_metrics["accuracy"].iloc[0]
        sk_acc = sk_eval[name]["accuracy"]
        print(f"  {name:35s} | sklearn acc={sk_acc:.3f}  xorq acc={xo_acc:.3f}")
        np.testing.assert_allclose(sk_acc, xo_acc, rtol=0.05)

    print("Assertions passed: sklearn and xorq accuracy values match.")

    # --- Plotting ---
    print("\n=== PLOTTING ===")

    # sklearn plot (eager, built in main)
    sk_fig = _plot_proba_grid(
        sk_results["classifiers_results"], class_names, "sklearn: "
    )

    # xorq deferred plot: fit on train_data (same data xorq_way fits on)
    xo_plot_fn = _build_proba_plot(
        class_names=tuple(class_names),
        feature_cols=feature_cols,
    )
    xo_png = deferred_matplotlib_plot(train_data, xo_plot_fn).execute()
    xo_img = load_plot_bytes(xo_png)

    # Composite side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(xo_img)
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(
        "Classification Probability: sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    out = "imgs/plot_classification_probability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
