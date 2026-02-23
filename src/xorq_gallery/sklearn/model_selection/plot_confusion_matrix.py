"""Confusion Matrix
==================

sklearn: Train SVC classifier on Iris dataset, compute confusion matrix on test set,
visualize both raw counts and normalized matrices via ConfusionMatrixDisplay.

xorq: Same SVC classifier wrapped in Pipeline.from_instance, fit/predict deferred,
compute confusion matrix via deferred aggregations, generate deferred confusion matrix
plots.

Both produce identical confusion matrices.

Dataset: Iris (sklearn)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
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

    return df, feature_cols, iris.target_names


def _build_pipeline():
    """Return sklearn Pipeline with StandardScaler and SVC."""
    return SklearnPipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)),
    ])


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_confusion_matrices(cm_raw, cm_norm, display_labels, title_prefix=""):
    """Plot both raw and normalized confusion matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Raw confusion matrix
    disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm_raw,
                                       display_labels=display_labels)
    disp_raw.plot(ax=axes[0], cmap="Blues", colorbar=True)
    axes[0].set_title(f"{title_prefix}Confusion Matrix (counts)")

    # Normalized confusion matrix
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm,
                                        display_labels=display_labels)
    disp_norm.plot(ax=axes[1], cmap="Blues", colorbar=True, values_format=".2f")
    axes[1].set_title(f"{title_prefix}Confusion Matrix (normalized)")

    plt.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit/predict, confusion matrix computation
# =========================================================================


def sklearn_way(df, feature_cols, target_names):
    """Eager sklearn: fit SVC on Iris dataset, compute confusion matrix.

    Returns dict with confusion matrices (raw and normalized) and test labels.
    """
    # Extract features and target
    X = df[feature_cols].values
    y = df["target"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    # Build and fit pipeline
    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Compute confusion matrices
    cm_raw = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

    print("sklearn confusion matrix (raw):")
    print(cm_raw)
    print("\nsklearn confusion matrix (normalized):")
    print(cm_norm)

    return {
        "cm_raw": cm_raw,
        "cm_norm": cm_norm,
        "y_test": y_test,
        "y_pred": y_pred,
        "pipeline": pipeline,
    }


# =========================================================================
# XORQ WAY -- deferred fit/predict, deferred confusion matrix computation
# =========================================================================


def xorq_way(df, feature_cols, target_names):
    """Deferred xorq: wrap SVC pipeline in Pipeline.from_instance, fit/predict
    deferred, compute confusion matrix via deferred aggregations.

    Returns dict with deferred expressions for confusion matrices and plot.
    """
    con = xo.connect()

    # Train/test split matching sklearn
    X = df[feature_cols].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    # Create dataframes
    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df["target"] = y_train
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df["target"] = y_test

    train_table = con.register(train_df, "train_iris")
    test_table = con.register(test_df, "test_iris")

    # Build xorq pipeline
    sklearn_pipeline = _build_pipeline()
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

    # Fit and predict (deferred)
    fitted = xorq_pipeline.fit(
        train_table,
        features=tuple(feature_cols),
        target="target"
    )
    preds = fitted.predict(test_table, name="pred")

    # Compute confusion matrix via deferred aggregations - flat, no loops
    # We need to compute cm[i,j] = count where target==i and pred==j
    # For a 3x3 matrix (Iris has 3 classes), we compute all 9 cells
    n_classes = len(target_names)

    # Raw confusion matrix: count occurrences - flat expansion of all 9 cells
    cm_raw_expr = preds.agg(
        cm_0_0=((preds.target == 0) & (preds.pred == 0)).sum(),
        cm_0_1=((preds.target == 0) & (preds.pred == 1)).sum(),
        cm_0_2=((preds.target == 0) & (preds.pred == 2)).sum(),
        cm_1_0=((preds.target == 1) & (preds.pred == 0)).sum(),
        cm_1_1=((preds.target == 1) & (preds.pred == 1)).sum(),
        cm_1_2=((preds.target == 1) & (preds.pred == 2)).sum(),
        cm_2_0=((preds.target == 2) & (preds.pred == 0)).sum(),
        cm_2_1=((preds.target == 2) & (preds.pred == 1)).sum(),
        cm_2_2=((preds.target == 2) & (preds.pred == 2)).sum(),
    )

    # Normalized confusion matrix: we'll compute it after executing raw matrix
    # For normalized, each row should sum to 1.0
    # This requires computing row sums first, then dividing

    # Deferred plot - we'll build it after executing the confusion matrices
    def _build_plot(cm_df, cm_raw_np, cm_norm_np, labels=target_names):
        """Build confusion matrix plots from materialized data."""
        fig = _plot_confusion_matrices(cm_raw_np, cm_norm_np, labels, "xorq: ")
        return fig

    # Store the plot building function and dependencies
    return {
        "cm_raw_expr": cm_raw_expr,
        "preds": preds,
        "n_classes": n_classes,
        "target_names": target_names,
        "_build_plot": _build_plot,
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df, feature_cols, target_names = _load_data()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(df, feature_cols, target_names)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(df, feature_cols, target_names)

    # Execute deferred confusion matrix
    print("\nExecuting xorq deferred expressions...")
    cm_raw_df = xo_results["cm_raw_expr"].execute()

    # Reconstruct confusion matrix from aggregated cells
    n_classes = xo_results["n_classes"]
    cm_raw_xo = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(n_classes):
        for j in range(n_classes):
            cell_name = f"cm_{i}_{j}"
            cm_raw_xo[i, j] = cm_raw_df[cell_name].iloc[0]

    print("\nxorq confusion matrix (raw):")
    print(cm_raw_xo)

    # Compute normalized version
    cm_norm_xo = cm_raw_xo.astype(float)
    row_sums = cm_norm_xo.sum(axis=1, keepdims=True)
    cm_norm_xo = cm_norm_xo / row_sums

    print("\nxorq confusion matrix (normalized):")
    print(cm_norm_xo)

    # ---- Assert numerical equivalence BEFORE plotting ----
    print("\n=== ASSERTIONS ===")
    np.testing.assert_array_equal(sk_results["cm_raw"], cm_raw_xo)
    print("Raw confusion matrix matches!")

    np.testing.assert_allclose(sk_results["cm_norm"], cm_norm_xo, rtol=1e-10)
    print("Normalized confusion matrix matches!")
    print("Assertions passed: sklearn and xorq confusion matrices match.")

    # Build sklearn plot
    sk_fig = _plot_confusion_matrices(
        sk_results["cm_raw"],
        sk_results["cm_norm"],
        target_names,
        "sklearn: "
    )

    # Build xorq plot
    xo_fig = xo_results["_build_plot"](cm_raw_df, cm_raw_xo, cm_norm_xo)

    # Composite side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    plt.suptitle("Confusion Matrix: sklearn vs xorq", fontsize=16)
    plt.tight_layout()
    out = "imgs/plot_confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
