"""Column Transformer with Mixed Types
===================================

sklearn: Build a ColumnTransformer that applies different preprocessing to numeric
(age, fare) and categorical (embarked, sex, pclass) features. Numeric features are
imputed and scaled; categorical features are one-hot encoded. Fit LogisticRegression,
evaluate accuracy.

xorq: Same ColumnTransformer pipeline wrapped in Pipeline.from_instance, deferred
fit/predict, accuracy via deferred_sklearn_metric.

Both produce identical results.

Dataset: Titanic (OpenML)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/compose/plot_column_transformer_mixed_types.py
"""

from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

Y_COL = "survived"
PRED_COL = "pred"
TEST_SIZE = 0.2
RANDOM_STATE = 0

NUMERIC_FEATURES = ("age", "fare")
CATEGORICAL_FEATURES = ("embarked", "sex", "pclass")
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Load Titanic dataset from OpenML."""
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    df = X[list(FEATURE_COLS)].copy()
    df["pclass"] = df["pclass"].astype(str)
    df[Y_COL] = y.astype(int)
    return df


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------


def _build_pipeline():
    """Build ColumnTransformer + LogisticRegression pipeline."""
    numeric_transformer = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(NUMERIC_FEATURES)),
            ("cat", categorical_transformer, list(CATEGORICAL_FEATURES)),
        ]
    )
    return SklearnPipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _confusion_matrix_figure(df, title):
    """Plot confusion matrix from predictions DataFrame with Y_COL and PRED_COL."""
    y_test = df[Y_COL].values
    y_pred = df[PRED_COL].values

    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    confusion = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(confusion, cmap="Blues", alpha=0.6)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", fontsize=20)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted 0", "Predicted 1"])
    ax.set_yticklabels(["Actual 0", "Actual 1"])
    ax.set_title(f"{title} (acc={accuracy:.4f})")
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


def plot_results(comparator):
    _, test_df = comparator.get_split_data()
    y_test = test_df[Y_COL].values

    sk_preds = comparator.sklearn_results[CT_LR_NAME]["preds"]
    xo_preds_df = comparator.xorq_results[CT_LR_NAME]["preds"]

    sk_fig = _confusion_matrix_figure(
        pd.DataFrame({Y_COL: y_test, PRED_COL: sk_preds}), title="sklearn"
    )
    xo_fig = _confusion_matrix_figure(xo_preds_df, title="xorq")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    fig.suptitle("Column Transformer with Mixed Types: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = (CT_LR_NAME,) = ("CT+LR",)
names_pipelines = ((CT_LR_NAME, _build_pipeline()),)
metrics_names_funcs = (("accuracy", accuracy_score),)

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=Y_COL,
    pred=PRED_COL,
    metrics_names_funcs=metrics_names_funcs,
    load_data=load_data,
    split_data=partial(
        train_test_split, test_size=TEST_SIZE, random_state=RANDOM_STATE
    ),
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)
# expose the exprs to invoke `xorq build plot_column_transformer_mixed_types.py --expr $expr_name`
(xorq_ct_lr_preds,) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    comparator.result_comparison
    save_fig("imgs/plot_column_transformer_mixed_types.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
