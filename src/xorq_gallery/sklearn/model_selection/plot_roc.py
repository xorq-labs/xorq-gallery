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
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toolz
import xorq.api as xo
from sklearn.datasets import load_iris
from sklearn.metrics import auc, roc_curve
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from xorq.expr.ml.metrics import deferred_auc_from_curve, deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import fig_to_image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 0


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Load Iris dataset and return as pandas DataFrame."""
    iris = load_iris()
    X, y = iris.data, iris.target

    # Create dataframe
    feature_cols = ("sepal_length", "sepal_width", "petal_length", "petal_width")
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y
    df["row_idx"] = range(len(df))

    n_classes = len(iris.target_names)

    return df, feature_cols, iris.target_names, n_classes


def _build_pipeline():
    """Return sklearn Pipeline with StandardScaler and SVC for binary classification."""
    return SklearnPipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)),
    ])


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_roc_curves(fpr_dict, tpr_dict, roc_auc_dict, n_classes, class_names, title_prefix=""):
    """Plot ROC curves for each class and micro-average."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.get_cmap("tab10")

    # Plot ROC curve for each class
    for i in range(n_classes):
        ax.plot(
            fpr_dict[i],
            tpr_dict[i],
            color=colors(i),
            lw=2,
            label=f"ROC curve of class {class_names[i]} (area = {roc_auc_dict[i]:.2f})",
        )

    # Plot micro-average ROC curve
    ax.plot(
        fpr_dict["micro"],
        tpr_dict["micro"],
        color="deeppink",
        linestyle=":",
        linewidth=4,
        label=f"micro-average ROC curve (area = {roc_auc_dict['micro']:.2f})",
    )

    # Plot diagonal
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


# =========================================================================
# SKLEARN WAY -- eager fit/predict_proba, ROC curve computation
# =========================================================================


def sklearn_way(train_df, test_df, feature_cols, class_names, n_classes):
    """Eager sklearn: fit binary classifiers per class, compute ROC curves.

    Returns dict with fpr, tpr, roc_auc for each class and micro-average.
    """
    X_train = train_df[list(feature_cols)].values
    y_train = train_df["target"].values
    X_test = test_df[list(feature_cols)].values
    y_test = test_df["target"].values

    # Binarize the output for One-vs-Rest
    y_train_bin = label_binarize(y_train, classes=range(n_classes))
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    # Fit binary classifiers for each class
    y_score = np.zeros((len(y_test), n_classes))
    for i in range(n_classes):
        pipeline = _build_pipeline()
        pipeline.fit(X_train, y_train_bin[:, i])
        y_score[:, i] = pipeline.predict_proba(X_test)[:, 1]

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(f"  sklearn: Class {class_names[i]:15s} | AUC = {roc_auc[i]:.4f}")

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print(f"  sklearn: {'micro-average':15s} | AUC = {roc_auc['micro']:.4f}")

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "y_test_bin": y_test_bin,
        "y_score": y_score,
    }


# =========================================================================
# XORQ WAY -- deferred fit/predict_proba, deferred ROC computation
# =========================================================================


def xorq_way(train_data, test_data, feature_cols, n_classes):
    """Deferred xorq: Pipeline.from_instance + predict_proba + deferred_auc_from_curve.

    Returns dict with deferred metric and proba expressions for each class.
    No .execute() is called here -- callers materialise when needed.
    """
    deferred = {}

    for i in range(n_classes):
        target_col_i = f"target_{i}"
        pred_col_i = f"scores_{i}"

        # Add binarized target column: 1 if target == i, else 0
        train_i = train_data.mutate(**{target_col_i: (train_data.target == i).cast(int)})
        test_i = test_data.mutate(**{target_col_i: (test_data.target == i).cast(int)})

        # Build fresh pipeline for each class
        pipeline = _build_pipeline()
        xorq_pipe = Pipeline.from_instance(pipeline)
        fitted = xorq_pipe.fit(train_i, features=tuple(feature_cols), target=target_col_i)
        proba_expr = fitted.predict_proba(test_i, name=pred_col_i)

        # Deferred AUC from ROC curve
        make_roc_auc = toolz.compose(
            deferred_auc_from_curve,
            deferred_sklearn_metric(target=target_col_i, pred=pred_col_i, metric=roc_curve),
        )
        metrics_expr = proba_expr.agg(**{f"roc_auc_{i}": make_roc_auc})

        deferred[i] = {
            "metrics_expr": metrics_expr,
            "proba_expr": proba_expr,
        }

    return deferred


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df, feature_cols, class_names, n_classes = _load_data()

    # Split using xorq train_test_splits
    con = xo.connect()
    table = con.register(df, "iris")
    train_data, test_data = xo.train_test_splits(
        table, test_sizes=0.25, unique_key="row_idx", random_seed=RANDOM_STATE
    )

    # Order by row_idx for reproducibility
    train_data = train_data.order_by("row_idx")
    test_data = test_data.order_by("row_idx")

    # Materialize for sklearn
    train_df = train_data.execute()
    test_df = test_data.execute()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(train_df, test_df, feature_cols, class_names, n_classes)

    print("\n=== XORQ WAY ===")
    xo_deferred = xorq_way(train_data, test_data, feature_cols, n_classes)

    # Materialise deferred expressions and extract ROC curves for plotting
    fpr_xo = {}
    tpr_xo = {}
    roc_auc_xo = {}

    for i in range(n_classes):
        metrics_df = xo_deferred[i]["metrics_expr"].execute()
        proba_df = xo_deferred[i]["proba_expr"].execute()

        auc_val = metrics_df[f"roc_auc_{i}"].iloc[0]
        roc_auc_xo[i] = auc_val
        print(f"  xorq:    Class {class_names[i]:15s} | AUC = {auc_val:.4f}")

        # predict_proba returns lists [p_class0, p_class1] per row; extract p_class1
        y_true = proba_df[f"target_{i}"].values
        y_score = np.array([p[1] for p in proba_df[f"scores_{i}"]])
        fpr_xo[i], tpr_xo[i], _ = roc_curve(y_true, y_score)

        # Stash materialised proba_df for micro-average computation below
        xo_deferred[i]["proba_df"] = proba_df

    # Compute micro-average
    y_test_all = np.column_stack([xo_deferred[i]["proba_df"][f"target_{i}"].values for i in range(n_classes)])
    y_score_all = np.column_stack([np.array([p[1] for p in xo_deferred[i]["proba_df"][f"scores_{i}"]]) for i in range(n_classes)])

    fpr_xo["micro"], tpr_xo["micro"], _ = roc_curve(y_test_all.ravel(), y_score_all.ravel())
    roc_auc_xo["micro"] = auc(fpr_xo["micro"], tpr_xo["micro"])
    print(f"  xorq:    {'micro-average':15s} | AUC = {roc_auc_xo['micro']:.4f}")

    # ---- Assert numerical equivalence BEFORE plotting ----
    print("\n=== ASSERTIONS ===")

    # Assert per-class AUC values match
    for i in range(n_classes):
        np.testing.assert_allclose(
            sk_results["roc_auc"][i],
            roc_auc_xo[i],
            rtol=0.05,
            err_msg=f"Class {i} AUC mismatch"
        )

    # Assert micro-average AUC matches
    np.testing.assert_allclose(
        sk_results["roc_auc"]["micro"],
        roc_auc_xo["micro"],
        rtol=0.05,
        err_msg="Micro-average AUC mismatch"
    )
    print("Assertions passed: sklearn and xorq ROC AUC values match.")

    # Build sklearn plot
    sk_fig = _plot_roc_curves(
        sk_results["fpr"],
        sk_results["tpr"],
        sk_results["roc_auc"],
        n_classes,
        class_names,
        "sklearn: "
    )

    # Build xorq plot
    xo_fig = _plot_roc_curves(
        fpr_xo,
        tpr_xo,
        roc_auc_xo,
        n_classes,
        class_names,
        "xorq: "
    )

    # Composite side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle("Multiclass ROC Curves: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    out = "imgs/plot_roc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
