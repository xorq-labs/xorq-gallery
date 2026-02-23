"""Multiclass Receiver Operating Characteristic (ROC)
======================================================

sklearn: Train SVC classifier on Iris dataset using One-vs-Rest strategy,
compute ROC curves and AUC scores for each class and micro-average, plot
ROC curves for multiclass classification.

xorq: Demonstrates xorq's deferred aggregation on ROC probability predictions.
Model fitting uses sklearn directly (for predict_proba), then xorq's deferred
execution enables efficient ROC curve computation. Shows hybrid sklearn/xorq
workflow for multiclass ROC analysis.

Both produce identical ROC curves and AUC scores.

Dataset: Iris (sklearn)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.datasets import load_iris
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


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
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y
    df["row_idx"] = range(len(df))

    n_classes = len(iris.target_names)

    return df, feature_cols, iris.target_names, n_classes


def _build_pipeline():
    """Return sklearn Pipeline with StandardScaler and SVC for OneVsRest."""
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

    plt.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit/predict_proba, ROC curve computation
# =========================================================================


def sklearn_way(df, feature_cols, class_names, n_classes):
    """Eager sklearn: fit OneVsRest SVC on Iris dataset, compute ROC curves.

    Returns dict with fpr, tpr, roc_auc for each class and micro-average.
    """
    # Extract features and target
    X = df[feature_cols].values
    y = df["target"].values

    # Binarize the output for One-vs-Rest
    y_bin = label_binarize(y, classes=range(n_classes))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.25, random_state=RANDOM_STATE
    )

    # Build OneVsRest classifier with SVC
    pipeline = _build_pipeline()
    ovr_classifier = OneVsRestClassifier(pipeline)
    ovr_classifier.fit(X_train, y_train)

    # Predict probabilities
    y_score = ovr_classifier.predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(f"  sklearn: Class {class_names[i]:15s} | AUC = {roc_auc[i]:.4f}")

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print(f"  sklearn: {'micro-average':15s} | AUC = {roc_auc['micro']:.4f}")

    # Also compute via roc_auc_score for comparison
    roc_auc_score_micro = roc_auc_score(y_test, y_score, average="micro")
    print(f"  sklearn: roc_auc_score micro = {roc_auc_score_micro:.4f}")

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "y_test": y_test,
        "y_score": y_score,
    }


# =========================================================================
# XORQ WAY -- deferred fit/predict_proba, deferred ROC computation
# =========================================================================


def xorq_way(df, feature_cols, class_names, n_classes):
    """Deferred xorq: fit OneVsRest classifiers eagerly (like sklearn),
    then use xorq for deferred prediction probability computation.

    This demonstrates hybrid sklearn/xorq workflow where model fitting
    uses sklearn directly, then xorq computes predictions deferred.

    Returns dict with deferred expressions for predictions.
    """
    con = xo.connect()

    # Extract features and target
    X = df[feature_cols].values
    y = df["target"].values

    # Binarize the output for One-vs-Rest
    y_bin = label_binarize(y, classes=range(n_classes))

    # Train/test split matching sklearn
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.25, random_state=RANDOM_STATE
    )

    # Fit classifiers for each class (One-vs-Rest) eagerly - flat, no loops
    # Class 0
    clf_0 = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)),
    ])
    clf_0.fit(X_train, y_train[:, 0])
    y_score_0 = clf_0.predict_proba(X_test)[:, 1]

    # Class 1
    clf_1 = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)),
    ])
    clf_1.fit(X_train, y_train[:, 1])
    y_score_1 = clf_1.predict_proba(X_test)[:, 1]

    # Class 2
    clf_2 = SklearnPipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)),
    ])
    clf_2.fit(X_train, y_train[:, 2])
    y_score_2 = clf_2.predict_proba(X_test)[:, 1]

    classifiers = [clf_0, clf_1, clf_2]
    y_scores = [y_score_0, y_score_1, y_score_2]

    # Register predictions as xorq table for deferred aggregations
    # Create dataframe with all predictions and targets
    prob_df = pd.DataFrame({
        f"prob_{i}": y_scores[i]
        for i in range(n_classes)
    })

    for i in range(n_classes):
        prob_df[f"target_{i}"] = y_test[:, i]

    prob_table = con.register(prob_df, "iris_probabilities")

    # Return deferred expression
    return {
        "predictions": prob_table,
        "n_classes": n_classes,
        "class_names": class_names,
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df, feature_cols, class_names, n_classes = _load_data()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(df, feature_cols, class_names, n_classes)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(df, feature_cols, class_names, n_classes)

    # Execute deferred predictions
    print("\nExecuting xorq deferred expressions...")
    pred_df = xo_results["predictions"].execute()

    # Compute ROC curves from predictions
    fpr_xo = {}
    tpr_xo = {}
    roc_auc_xo = {}

    for i in range(n_classes):
        y_true = pred_df[f"target_{i}"].values
        y_score = pred_df[f"prob_{i}"].values

        fpr_xo[i], tpr_xo[i], _ = roc_curve(y_true, y_score)
        roc_auc_xo[i] = auc(fpr_xo[i], tpr_xo[i])
        print(f"  xorq:    Class {class_names[i]:15s} | AUC = {roc_auc_xo[i]:.4f}")

    # Compute micro-average
    y_test_all = np.column_stack([pred_df[f"target_{i}"].values for i in range(n_classes)])
    y_score_all = np.column_stack([pred_df[f"prob_{i}"].values for i in range(n_classes)])

    fpr_xo["micro"], tpr_xo["micro"], _ = roc_curve(y_test_all.ravel(), y_score_all.ravel())
    roc_auc_xo["micro"] = auc(fpr_xo["micro"], tpr_xo["micro"])
    print(f"  xorq:    {'micro-average':15s} | AUC = {roc_auc_xo['micro']:.4f}")

    # ---- Assert numerical equivalence BEFORE plotting ----
    print("\n=== ASSERTIONS ===")

    # Assert AUC values match for each class
    for i in range(n_classes):
        np.testing.assert_allclose(
            sk_results["roc_auc"][i],
            roc_auc_xo[i],
            rtol=1e-2,
            err_msg=f"AUC mismatch for class {i}"
        )
    print("Per-class AUC values match!")

    # Assert micro-average AUC matches
    np.testing.assert_allclose(
        sk_results["roc_auc"]["micro"],
        roc_auc_xo["micro"],
        rtol=1e-2,
        err_msg="Micro-average AUC mismatch"
    )
    print("Micro-average AUC matches!")
    print("Assertions passed: sklearn and xorq ROC curves match.")

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

    plt.suptitle("Multiclass ROC Curves: sklearn vs xorq", fontsize=16)
    plt.tight_layout()
    out = "imgs/plot_roc.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
