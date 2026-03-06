"""Confusion Matrix
==================

sklearn: Train SVC classifier on Iris dataset, compute confusion matrix on test set,
visualize both raw counts and normalized matrices via ConfusionMatrixDisplay.

xorq: Same SVC classifier wrapped in Pipeline.from_instance, fit/predict deferred,
accuracy via deferred_sklearn_metric, confusion matrices match sklearn.

Both produce identical confusion matrices.

Dataset: Iris (sklearn)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/model_selection/plot_confusion_matrix.py
"""

from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
)
from xorq_gallery.utils import (
    fig_to_image,
    save_fig,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 0
TEST_SIZE = 0.25
TARGET_COL = "target"
PRED_COL = "pred"
FEATURE_COLS = ("sepal_length", "sepal_width", "petal_length", "petal_width")
TARGET_NAMES = ("setosa", "versicolor", "virginica")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Load Iris dataset as DataFrame."""
    iris = load_iris()
    return pd.DataFrame(iris.data, columns=FEATURE_COLS).assign(
        **{TARGET_COL: iris.target}
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_confusion_matrices(cm_raw, cm_norm, display_labels, title_prefix=""):
    """Plot both raw and normalized confusion matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay(confusion_matrix=cm_raw, display_labels=display_labels).plot(
        ax=axes[0], cmap="Blues", colorbar=True
    )
    axes[0].set_title(f"{title_prefix}Confusion Matrix (counts)")

    ConfusionMatrixDisplay(
        confusion_matrix=cm_norm, display_labels=display_labels
    ).plot(ax=axes[1], cmap="Blues", colorbar=True, values_format=".2f")
    axes[1].set_title(f"{title_prefix}Confusion Matrix (normalized)")

    fig.tight_layout()
    return fig


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
        print(f"  {name} accuracy - sklearn: {sk_acc:.4f}, xorq: {xo_acc:.4f}")

    # Assert confusion matrices match
    _, test_df = comparator.get_split_data()
    y_test = test_df[TARGET_COL].values
    sk_cm = confusion_matrix(y_test, sklearn_results[SVC_NAME]["preds"])
    xo_cm = confusion_matrix(
        y_test, comparator.xorq_results[SVC_NAME]["preds"][PRED_COL].values
    )
    np.testing.assert_array_equal(sk_cm, xo_cm)
    print("Confusion matrices match.")


def plot_results(comparator):
    _, test_df = comparator.get_split_data()
    y_test = test_df[TARGET_COL].values

    sk_preds = comparator.sklearn_results[SVC_NAME]["preds"]
    xo_preds = comparator.xorq_results[SVC_NAME]["preds"][PRED_COL].values

    sk_cm = confusion_matrix(y_test, sk_preds)
    sk_cm_norm = confusion_matrix(y_test, sk_preds, normalize="true")
    xo_cm = confusion_matrix(y_test, xo_preds)
    xo_cm_norm = confusion_matrix(y_test, xo_preds, normalize="true")

    sk_fig = _plot_confusion_matrices(sk_cm, sk_cm_norm, TARGET_NAMES, "sklearn: ")
    xo_fig = _plot_confusion_matrices(xo_cm, xo_cm_norm, TARGET_NAMES, "xorq: ")

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    fig.suptitle("Confusion Matrix: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = (SVC_NAME,) = ("SVC",)
names_pipelines = (
    (
        SVC_NAME,
        SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)),
            ]
        ),
    ),
)
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
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)
# expose the exprs to invoke `xorq build plot_confusion_matrix.py --expr $expr_name`
(xorq_svc_preds,) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    comparator.result_comparison
    save_fig("imgs/plot_confusion_matrix.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
