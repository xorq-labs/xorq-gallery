"""Multiclass Receiver Operating Characteristic (ROC)
======================================================

sklearn: Train SVC classifier on Iris dataset using One-vs-Rest strategy,
compute ROC curves and AUC scores for each class and micro-average, plot
ROC curves for multiclass classification.

xorq: Uses deferred execution via Pipeline.from_instance, fit, predict_proba,
and deferred_auc_from_curve for each binary classifier. All expressions are
built lazily; execution is deferred to the caller.

Both produce equivalent ROC curves and AUC scores.

Dataset: Iris (sklearn)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/model_selection/plot_roc.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toolz
import xorq.api as xo
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.metrics import auc, roc_curve
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xorq.expr.ml.metrics import deferred_auc_from_curve, deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
)
from xorq_gallery.utils import fig_to_image, save_fig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 0
FEATURE_COLS = ("sepal_length", "sepal_width", "petal_length", "petal_width")
ROW_IDX = "row_idx"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_iris():
    """Load Iris dataset and return as pandas DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=FEATURE_COLS)
    df["target"] = iris.target
    df[ROW_IDX] = range(len(df))
    return df, iris.target_names, len(iris.target_names)


def _build_pipeline():
    """SVC pipeline for binary classification (OvR)."""
    return SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)),
        ]
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_roc_curves(
    fpr_dict, tpr_dict, roc_auc_dict, n_classes, class_names, title_prefix=""
):
    """Plot ROC curves for each class and micro-average."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.get_cmap("tab10")

    for i in range(n_classes):
        ax.plot(
            fpr_dict[i],
            tpr_dict[i],
            color=colors(i),
            lw=2,
            label=f"ROC curve of class {class_names[i]} (area = {roc_auc_dict[i]:.2f})",
        )

    ax.plot(
        fpr_dict["micro"],
        tpr_dict["micro"],
        color="deeppink",
        linestyle=":",
        linewidth=4,
        label=f"micro-average ROC curve (area = {roc_auc_dict['micro']:.2f})",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Chance level (AUC = 0.5)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title_prefix}ROC Curves - One-vs-Rest multiclass")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Custom make_*_result — per-class binary OvR with predict_proba
# ---------------------------------------------------------------------------


def _make_sklearn_result_for_class(class_idx):
    """Return a make_sklearn_result for one binary OvR class."""

    def _make(pipeline, train_data, test_data, features, target, metrics_names_funcs):
        X_train = train_data[list(features)].values
        y_train_bin = (train_data["target"].values == class_idx).astype(int)
        X_test = test_data[list(features)].values
        y_test_bin = (test_data["target"].values == class_idx).astype(int)

        fitted = clone(pipeline).fit(X_train, y_train_bin)
        y_score = fitted.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_bin, y_score)
        roc_auc_val = auc(fpr, tpr)

        return {
            "fitted": fitted.steps[-1][-1],
            "preds": y_score,
            "metrics": {"roc_auc": roc_auc_val},
            "other": {
                "fpr": fpr,
                "tpr": tpr,
                "y_test_bin": y_test_bin,
                "y_score": y_score,
            },
        }

    return _make


def _make_deferred_xorq_result_for_class(class_idx):
    """Return a make_deferred_xorq_result for one binary OvR class."""

    target_col_i = f"target_{class_idx}"
    pred_col_i = f"scores_{class_idx}"

    def _make(
        pipeline, train_data, test_data, features, target, metrics_names_funcs, pred
    ):
        train_i = train_data.mutate(
            **{target_col_i: (train_data.target == class_idx).cast(int)}
        )
        test_i = test_data.mutate(
            **{target_col_i: (test_data.target == class_idx).cast(int)}
        )

        xorq_fitted = Pipeline.from_instance(pipeline).fit(
            train_i, features=tuple(features), target=target_col_i
        )
        proba_expr = xorq_fitted.predict_proba(test_i, name=pred_col_i)

        make_roc_auc = toolz.compose(
            deferred_auc_from_curve,
            deferred_sklearn_metric(
                target=target_col_i, pred=pred_col_i, metric=roc_curve
            ),
        )
        metrics = {
            f"roc_auc_{class_idx}": proba_expr.agg(
                **{f"roc_auc_{class_idx}": make_roc_auc}
            )
        }

        return {
            "xorq_fitted": xorq_fitted,
            "preds": proba_expr,
            "metrics": metrics,
            "other": {
                "target_col": lambda: target_col_i,
                "pred_col": lambda: pred_col_i,
            },
        }

    return _make


def _make_xorq_result_for_class(class_idx):
    """Return a make_xorq_result for one binary OvR class."""

    target_col_i = f"target_{class_idx}"
    pred_col_i = f"scores_{class_idx}"

    def _make(deferred_xorq_result):
        xorq_fitted = deferred_xorq_result["xorq_fitted"]
        preds_df = deferred_xorq_result["preds"].execute()

        metric_key = f"roc_auc_{class_idx}"
        metrics_df = deferred_xorq_result["metrics"][metric_key].execute()
        roc_auc_val = metrics_df[metric_key].iloc[0]

        y_test_bin = preds_df[target_col_i].values
        y_score = np.array([p[1] for p in preds_df[pred_col_i]])
        fpr, tpr, _ = roc_curve(y_test_bin, y_score)

        return {
            "fitted": xorq_fitted.fitted_steps[-1].model,
            "preds": y_score,
            "metrics": {"roc_auc": roc_auc_val},
            "other": {
                "fpr": fpr,
                "tpr": tpr,
                "y_test_bin": y_test_bin,
                "y_score": y_score,
            },
        }

    return _make


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def _noop_compare(comparator):
    pass


def _noop_plot(comparator):
    return None


# ---------------------------------------------------------------------------
# Split via xorq train_test_splits
# ---------------------------------------------------------------------------


def split_data(df):
    con = xo.connect()
    table = con.register(df, "iris_roc")
    train_data, test_data = xo.train_test_splits(
        table, test_sizes=0.25, unique_key=ROW_IDX, random_seed=RANDOM_STATE
    )
    return (
        train_data.order_by(ROW_IDX).execute(),
        test_data.order_by(ROW_IDX).execute(),
    )


# ---------------------------------------------------------------------------
# Module-level setup — one comparator per class
# ---------------------------------------------------------------------------

_iris_df, class_names, n_classes = _load_iris()

# Single pipeline tuple used for all classes
_pipeline = _build_pipeline()
_names_pipelines = (("SVC", _pipeline),)

comparators = {
    i: SklearnXorqComparator(
        names_pipelines=_names_pipelines,
        features=FEATURE_COLS,
        target="target",
        pred=f"scores_{i}",
        metrics_names_funcs=(),
        load_data=lambda: _iris_df,
        split_data=split_data,
        make_sklearn_result=_make_sklearn_result_for_class(i),
        make_deferred_xorq_result=_make_deferred_xorq_result_for_class(i),
        make_xorq_result=_make_xorq_result_for_class(i),
        compare_results_fn=_noop_compare,
        plot_results_fn=_noop_plot,
    )
    for i in range(n_classes)
}

# Module-level deferred exprs (one per class)
(xorq_class0_proba, xorq_class1_proba, xorq_class2_proba) = (
    comparators[i].deferred_xorq_results["SVC"]["preds"] for i in range(n_classes)
)


# =========================================================================
# Run
# =========================================================================


def main():
    # Collect results for all classes from comparators
    fpr_sk, tpr_sk, roc_auc_sk = {}, {}, {}
    fpr_xo, tpr_xo, roc_auc_xo = {}, {}, {}
    y_test_bins_sk, y_scores_sk = [], []
    y_test_bins_xo, y_scores_xo = [], []

    print("=== SKLEARN WAY ===")
    for i in range(n_classes):
        sk = comparators[i].sklearn_results["SVC"]
        fpr_sk[i] = sk["other"]["fpr"]
        tpr_sk[i] = sk["other"]["tpr"]
        roc_auc_sk[i] = sk["metrics"]["roc_auc"]
        y_test_bins_sk.append(sk["other"]["y_test_bin"])
        y_scores_sk.append(sk["other"]["y_score"])
        print(f"  sklearn: Class {class_names[i]:15s} | AUC = {roc_auc_sk[i]:.4f}")

    print("\n=== XORQ WAY ===")
    for i in range(n_classes):
        xo_r = comparators[i].xorq_results["SVC"]
        fpr_xo[i] = xo_r["other"]["fpr"]
        tpr_xo[i] = xo_r["other"]["tpr"]
        roc_auc_xo[i] = xo_r["metrics"]["roc_auc"]
        y_test_bins_xo.append(xo_r["other"]["y_test_bin"])
        y_scores_xo.append(xo_r["other"]["y_score"])
        print(f"  xorq:    Class {class_names[i]:15s} | AUC = {roc_auc_xo[i]:.4f}")

    # Micro-average
    fpr_sk["micro"], tpr_sk["micro"], _ = roc_curve(
        np.concatenate(y_test_bins_sk), np.concatenate(y_scores_sk)
    )
    roc_auc_sk["micro"] = auc(fpr_sk["micro"], tpr_sk["micro"])

    fpr_xo["micro"], tpr_xo["micro"], _ = roc_curve(
        np.concatenate(y_test_bins_xo), np.concatenate(y_scores_xo)
    )
    roc_auc_xo["micro"] = auc(fpr_xo["micro"], tpr_xo["micro"])

    # Assertions
    print("\n=== ASSERTIONS ===")
    for i in range(n_classes):
        np.testing.assert_allclose(roc_auc_sk[i], roc_auc_xo[i], rtol=0.05)
    np.testing.assert_allclose(roc_auc_sk["micro"], roc_auc_xo["micro"], rtol=0.05)
    print("Assertions passed: sklearn and xorq ROC AUC values match.")

    # Plot
    sk_fig = _plot_roc_curves(
        fpr_sk, tpr_sk, roc_auc_sk, n_classes, class_names, "sklearn: "
    )
    xo_fig = _plot_roc_curves(
        fpr_xo, tpr_xo, roc_auc_xo, n_classes, class_names, "xorq: "
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle("Multiclass ROC Curves: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    save_fig("imgs/plot_roc.png", fig)

    plt.close(sk_fig)
    plt.close(xo_fig)


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
