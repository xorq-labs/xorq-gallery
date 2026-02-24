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
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from toolz import curry
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Constants and feature groups
# ---------------------------------------------------------------------------

Y_COL = "survived"
PRED_COL = "pred"
ROW_IDX = "row_idx"
TEST_SIZE = 0.2
RANDOM_STATE = 0

NUMERIC_FEATURES = ("age", "fare")
CATEGORICAL_FEATURES = ("embarked", "sex", "pclass")
ALL_FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data():
    """Load Titanic dataset from OpenML."""
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

    df = X.copy()
    df[Y_COL] = y

    # Select only the features we need
    df = df[list(ALL_FEATURE_COLS) + [Y_COL]]

    # Convert pclass to string (categorical) for consistency
    df["pclass"] = df["pclass"].astype(str)

    # Convert target to int (0 or 1)
    df[Y_COL] = df[Y_COL].astype(int)

    df[ROW_IDX] = range(len(df))

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
# Plotting helper
# ---------------------------------------------------------------------------


@curry
def _confusion_matrix_figure(df, title):
    """Plot confusion matrix from predictions DataFrame."""
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


# =========================================================================
# SKLEARN WAY -- eager fit/predict
# =========================================================================


def sklearn_way(train_df, test_df, clf):
    """Eager sklearn: fit on train, predict/score on test."""
    X_train = train_df[list(ALL_FEATURE_COLS)]
    y_train = train_df[Y_COL]
    X_test = test_df[list(ALL_FEATURE_COLS)]
    y_test = test_df[Y_COL]

    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)

    print(f"  sklearn accuracy: {accuracy:.4f}")

    return {
        "accuracy": accuracy,
        "y_test": y_test.values,
        "y_pred": y_pred,
    }


# =========================================================================
# XORQ WAY -- deferred fit/predict
# =========================================================================


def xorq_way(train_data, test_data, clf):
    """Deferred xorq: Pipeline.from_instance, deferred fit/predict/score.

    Returns deferred predictions and metrics.
    Nothing is executed until .execute().
    """
    xorq_pipe = Pipeline.from_instance(clf)
    fitted = xorq_pipe.fit(train_data, features=ALL_FEATURE_COLS, target=Y_COL)
    preds = fitted.predict(test_data, name=PRED_COL)

    make_metric = deferred_sklearn_metric(target=Y_COL, pred=PRED_COL)
    metrics = preds.agg(accuracy=make_metric(metric=accuracy_score))

    return {
        "predictions": preds,
        "metrics": metrics,
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    print("Loading data...")
    df = _load_data()

    # Hash-based split via xorq -- single source of truth for both paths
    con = xo.connect()
    table = con.register(df, "titanic_data")
    train_data, test_data = xo.train_test_splits(
        table,
        test_sizes=TEST_SIZE,
        unique_key=ROW_IDX,
        random_seed=RANDOM_STATE,
    )

    # Sort for deterministic ordering
    train_data = train_data.order_by(ROW_IDX)
    test_data = test_data.order_by(ROW_IDX)

    # Materialize for sklearn
    train_df = train_data.execute()
    test_df = test_data.execute()

    print("=== SKLEARN WAY ===")
    sk_clf = _build_pipeline()
    sk_results = sklearn_way(train_df, test_df, sk_clf)

    print("\n=== XORQ WAY ===")
    xo_clf = _build_pipeline()
    xo_results = xorq_way(train_data, test_data, xo_clf)

    # Execute deferred expressions
    metrics_df = xo_results["metrics"].execute()
    xo_accuracy = metrics_df["accuracy"].iloc[0]
    print(f"  xorq   accuracy: {xo_accuracy:.4f}")

    # Assert
    print("\n=== ASSERTIONS ===")
    sk_acc_df = pd.DataFrame({"accuracy": [sk_results["accuracy"]]})
    xo_acc_df = pd.DataFrame({"accuracy": [xo_accuracy]})
    pd.testing.assert_frame_equal(sk_acc_df, xo_acc_df, rtol=1e-9)
    print("Assertions passed: sklearn and xorq accuracies match.")

    # Plot
    print("\n=== PLOTTING ===")
    xo_preds_executed = (
        xo_results["predictions"].execute().sort_values(ROW_IDX).reset_index(drop=True)
    )

    # sklearn confusion matrix
    sk_fig = _confusion_matrix_figure(
        pd.DataFrame({Y_COL: sk_results["y_test"], PRED_COL: sk_results["y_pred"]}),
        title="sklearn",
    )

    # xorq deferred confusion matrix
    xo_plot_table = con.register(
        xo_preds_executed[[Y_COL, PRED_COL]],
        "titanic_preds",
    )
    xo_png = deferred_matplotlib_plot(
        xo_plot_table,
        _confusion_matrix_figure(title="xorq"),
    ).execute()

    xo_img = load_plot_bytes(xo_png)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(xo_img)
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(
        "Column Transformer with Mixed Types: sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    out = "imgs/column_transformer_mixed_types.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Composite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
