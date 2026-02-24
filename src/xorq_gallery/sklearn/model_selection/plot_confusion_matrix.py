"""Confusion Matrix
==================

sklearn: Train SVC classifier on Iris dataset, compute confusion matrix on test set,
visualize both raw counts and normalized matrices via ConfusionMatrixDisplay.

xorq: Same SVC classifier wrapped in Pipeline.from_instance, fit/predict deferred,
compute confusion matrix via deferred_sklearn_metric(metric=confusion_matrix).

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
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import fig_to_image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 0
TEST_SIZE = 0.25
TARGET_COL = "target"
PRED_COL = "pred"
ROW_IDX = "row_idx"


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Load Iris dataset and return as pandas DataFrame."""
    iris = load_iris()
    X, y = iris.data, iris.target

    feature_cols = ("sepal_length", "sepal_width", "petal_length", "petal_width")
    df = pd.DataFrame(X, columns=feature_cols)
    df[TARGET_COL] = y
    df[ROW_IDX] = range(len(df))

    return df, feature_cols, iris.target_names


def _build_pipeline():
    """Return sklearn Pipeline with StandardScaler and SVC."""
    return SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)),
        ]
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_confusion_matrices(cm_raw, cm_norm, display_labels, title_prefix=""):
    """Plot both raw and normalized confusion matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    disp_raw = ConfusionMatrixDisplay(
        confusion_matrix=cm_raw, display_labels=display_labels
    )
    disp_raw.plot(ax=axes[0], cmap="Blues", colorbar=True)
    axes[0].set_title(f"{title_prefix}Confusion Matrix (counts)")

    disp_norm = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm, display_labels=display_labels
    )
    disp_norm.plot(ax=axes[1], cmap="Blues", colorbar=True, values_format=".2f")
    axes[1].set_title(f"{title_prefix}Confusion Matrix (normalized)")

    fig.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit/predict, confusion matrix computation
# =========================================================================


def sklearn_way(train_df, test_df, feature_cols, clf):
    """Eager sklearn: fit SVC on train, predict on test, compute confusion matrix."""
    X_train = train_df[list(feature_cols)]
    y_train = train_df[TARGET_COL]
    X_test = test_df[list(feature_cols)]
    y_test = test_df[TARGET_COL]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm_raw = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

    print(f"  sklearn accuracy: {(y_pred == y_test.values).mean():.4f}")

    return {
        "cm_raw": cm_raw,
        "cm_norm": cm_norm,
    }


# =========================================================================
# XORQ WAY -- deferred fit/predict, deferred confusion matrix
# =========================================================================


def xorq_way(train_data, test_data, feature_cols, clf):
    """Deferred xorq: Pipeline.from_instance, deferred fit/predict,
    confusion matrix via deferred_sklearn_metric.

    Returns deferred predictions and confusion matrix metric expression.
    """
    xorq_pipe = Pipeline.from_instance(clf)
    fitted = xorq_pipe.fit(train_data, features=tuple(feature_cols), target=TARGET_COL)
    preds = fitted.predict(test_data, name=PRED_COL)

    make_metric = deferred_sklearn_metric(target=TARGET_COL, pred=PRED_COL)
    metrics = preds.agg(cm=make_metric(metric=confusion_matrix))

    return {
        "predictions": preds,
        "metrics": metrics,
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df, feature_cols, target_names = _load_data()

    # Hash-based split via xorq -- single source of truth for both paths
    con = xo.connect()
    table = con.register(df, "iris_data")
    train_data, test_data = xo.train_test_splits(
        table,
        test_sizes=TEST_SIZE,
        unique_key=ROW_IDX,
        random_seed=RANDOM_STATE,
    )
    train_data = train_data.order_by(ROW_IDX)
    test_data = test_data.order_by(ROW_IDX)

    # Materialize for sklearn
    train_df = train_data.execute()
    test_df = test_data.execute()

    print("=== SKLEARN WAY ===")
    sk_clf = _build_pipeline()
    sk_results = sklearn_way(train_df, test_df, feature_cols, sk_clf)

    print("\n=== XORQ WAY ===")
    xo_clf = _build_pipeline()
    xo_results = xorq_way(train_data, test_data, feature_cols, xo_clf)

    # Execute deferred confusion matrix
    metrics_df = xo_results["metrics"].execute()
    cm_raw_xo = np.array(metrics_df["cm"].iloc[0])
    cm_norm_xo = cm_raw_xo.astype(float) / cm_raw_xo.sum(axis=1, keepdims=True)

    print(f"  xorq   accuracy: {np.trace(cm_raw_xo) / cm_raw_xo.sum():.4f}")

    # Assert
    print("\n=== ASSERTIONS ===")
    np.testing.assert_array_equal(sk_results["cm_raw"], cm_raw_xo)
    print("Raw confusion matrices match.")
    np.testing.assert_allclose(sk_results["cm_norm"], cm_norm_xo, rtol=1e-10)
    print("Normalized confusion matrices match.")
    print("Assertions passed.")

    # Plot
    print("\n=== PLOTTING ===")
    sk_fig = _plot_confusion_matrices(
        sk_results["cm_raw"], sk_results["cm_norm"], target_names, "sklearn: "
    )
    xo_fig = _plot_confusion_matrices(cm_raw_xo, cm_norm_xo, target_names, "xorq: ")

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle("Confusion Matrix: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    out = "imgs/plot_confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
