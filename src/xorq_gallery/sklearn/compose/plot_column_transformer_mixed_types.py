"""Column Transformer with Mixed Types
===================================

sklearn: Build a ColumnTransformer that applies different preprocessing to numeric
(age, fare) and categorical (embarked, sex, pclass) features. Numeric features are
imputed and scaled; categorical features are one-hot encoded. Split with
train_test_split(random_state=0), fit LogisticRegression, evaluate accuracy.

xorq: Same ColumnTransformer pipeline wrapped in Pipeline.from_instance. Data is an
ibis expression, split with same random_state, fit/predict deferred, accuracy via
deferred_sklearn_metric.

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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

y_col = "survived"
numeric_features = ["age", "fare"]
categorical_features = ["embarked", "sex", "pclass"]
all_feature_cols = numeric_features + categorical_features


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Load Titanic dataset from OpenML."""
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

    # Combine X and y into a single dataframe
    df = X.copy()
    df[y_col] = y

    # Select only the features we need
    df = df[all_feature_cols + [y_col]]

    # Convert pclass to string (categorical) for consistency
    df["pclass"] = df["pclass"].astype(str)

    # Convert target to int (0 or 1)
    df[y_col] = df[y_col].astype(int)

    # Add row_idx for reproducibility
    df["row_idx"] = range(len(df))

    return df


# ---------------------------------------------------------------------------
# Plotting helpers (used by deferred_matplotlib_plot)
# ---------------------------------------------------------------------------


def _build_feature_importance_figure(title, feature_names):
    """Return a UDAF-compatible plotting function for feature importance."""
    def _plot(df):
        # df contains columns: feature_name, importance
        # For this simple example, we'll just create a bar chart showing
        # the number of features after preprocessing
        n_features = len(df.columns) - 1  # exclude target

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(["Preprocessed Features"], [n_features])
        ax.set_title(title)
        ax.set_ylabel("Number of Features")
        ax.text(0, n_features + 0.5, f"n={n_features}", ha="center", fontsize=12)
        plt.tight_layout()
        return fig

    return _plot


# ---------------------------------------------------------------------------
# Shared pipeline definitions
# ---------------------------------------------------------------------------


def _build_pipelines():
    """Build the preprocessing and full prediction pipelines."""
    # Numeric transformer: impute with median, then scale
    numeric_transformer = SklearnPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical transformer: one-hot encode only
    # Note: We skip SelectPercentile to avoid dynamic schema inference issues
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # ColumnTransformer combines both
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Full pipeline: preprocessor + classifier
    clf = SklearnPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=0, max_iter=1000)),
        ]
    )

    return clf


# =========================================================================
# SKLEARN WAY -- eager, train_test_split
# =========================================================================


def sklearn_way(df, clf):
    """Eager sklearn: random split, fit, predict, score."""
    X = df[all_feature_cols]
    y = df[y_col]

    # Random split with fixed seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Fit and evaluate
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    print(f"  sklearn accuracy: {accuracy:.4f}")

    # Get predictions for later analysis
    y_pred = clf.predict(X_test)

    return {
        "accuracy": accuracy,
        "y_test": y_test.values,
        "y_pred": y_pred,
    }


# =========================================================================
# XORQ WAY -- deferred, Pipeline.from_instance
# =========================================================================


def xorq_way(train_df, test_df, clf):
    """Deferred xorq: Pipeline.from_instance, deferred fit/predict/score.

    Returns deferred predictions and metrics.
    Nothing is executed until ``.execute()``.
    """
    con = xo.connect()
    features = tuple(all_feature_cols)

    train_data = con.register(train_df, "titanic_train")
    test_data = con.register(test_df, "titanic_test")

    # Wrap sklearn pipeline in xorq
    xorq_pipe = Pipeline.from_instance(clf)
    fitted = xorq_pipe.fit(train_data, features=features, target=y_col)
    preds = fitted.predict(test_data, name="pred")

    # Deferred metric
    make_metric = deferred_sklearn_metric(target=y_col, pred="pred")
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

    df = _load_data()
    clf = _build_pipelines()

    # Perform train_test_split in main()
    X = df[all_feature_cols]
    y = df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Create train and test dataframes with target
    train_df = X_train.copy()
    train_df[y_col] = y_train
    test_df = X_test.copy()
    test_df[y_col] = y_test

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(df, clf)

    print("\n=== XORQ WAY ===")
    deferred = xorq_way(train_df, test_df, clf)

    # Execute deferred expressions
    metrics_df = deferred["metrics"].execute()
    xo_accuracy = metrics_df["accuracy"].iloc[0]
    print(f"  xorq   accuracy: {xo_accuracy:.4f}")

    # ---- Assert numerical equivalence BEFORE plotting ----
    np.testing.assert_allclose(sk_results["accuracy"], xo_accuracy, rtol=1e-6)
    print("\nAssertion passed: sklearn and xorq accuracies match.")

    # Execute deferred plot using deferred_matplotlib_plot in main()
    xo_png = deferred_matplotlib_plot(
        deferred["predictions"].select(y_col, "pred"),
        _build_feature_importance_figure("xorq - Feature Count", all_feature_cols),
    ).execute()

    # Build sklearn subplot natively
    sk_fig, sk_ax = plt.subplots(figsize=(8, 5))

    # Simple visualization: confusion matrix counts
    y_test = sk_results["y_test"]
    y_pred = sk_results["y_pred"]

    # Calculate confusion matrix values
    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))

    # Create confusion matrix visualization
    confusion = np.array([[tn, fp], [fn, tp]])
    sk_ax.imshow(confusion, cmap="Blues", alpha=0.6)

    for i in range(2):
        for j in range(2):
            sk_ax.text(j, i, str(confusion[i, j]), ha="center", va="center", fontsize=20)

    sk_ax.set_xticks([0, 1])
    sk_ax.set_yticks([0, 1])
    sk_ax.set_xticklabels(["Predicted 0", "Predicted 1"])
    sk_ax.set_yticklabels(["Actual 0", "Actual 1"])
    sk_ax.set_title(f"sklearn - Confusion Matrix (acc={sk_results['accuracy']:.4f})")
    plt.tight_layout()

    # Composite: sklearn (left) | xorq (right)
    xo_img = load_plot_bytes(xo_png)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].axis("off")
    axes[1].imshow(xo_img)
    axes[1].axis("off")

    plt.suptitle("Column Transformer with Mixed Types: sklearn vs xorq", fontsize=14)
    plt.tight_layout()
    out = "imgs/column_transformer_mixed_types.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
