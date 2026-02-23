"""Classification of text documents using sparse features
===========================================================

sklearn: Load 20 newsgroups dataset (4 categories), vectorize with TfidfVectorizer
(max_df=0.5, min_df=5, sublinear_tf=True), train RidgeClassifier on training data,
predict on test data, compute accuracy and confusion matrix.

xorq: Same preprocessing pipeline wrapped in Pipeline.from_instance, deferred
fit/predict on registered tables, deferred accuracy metric, deferred confusion matrix plot.

Both produce identical accuracy and confusion matrices.

Dataset: 20 newsgroups (4 categories: alt.atheism, talk.religion.misc, comp.graphics, sci.space)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.pipeline import Pipeline as SklearnPipeline
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

CATEGORIES = [
    "alt.atheism",
    "talk.religion.misc",
    "comp.graphics",
    "sci.space",
]

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Load and vectorize the 20 newsgroups dataset.

    Returns a dict with train/test splits already vectorized, along with
    target names for plotting.
    """
    # Fetch train data
    data_train = fetch_20newsgroups(
        subset="train",
        categories=CATEGORIES,
        shuffle=True,
        random_state=RANDOM_STATE,
        remove=("headers", "footers", "quotes"),  # Strip metadata
    )

    # Fetch test data
    data_test = fetch_20newsgroups(
        subset="test",
        categories=CATEGORIES,
        shuffle=True,
        random_state=RANDOM_STATE,
        remove=("headers", "footers", "quotes"),
    )

    # Vectorize
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
    )
    X_train = vectorizer.fit_transform(data_train.data)
    X_test = vectorizer.transform(data_test.data)

    y_train = data_train.target
    y_test = data_test.target
    target_names = data_train.target_names

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "target_names": target_names,
    }


def _build_pipeline():
    """Return sklearn Pipeline with RidgeClassifier.

    Note: We don't wrap TfidfVectorizer in the pipeline here because:
    1. The sklearn example vectorizes separately
    2. For xorq, we need dense features as input (vectorization happens before)
    """
    return SklearnPipeline([
        ("clf", RidgeClassifier(tol=1e-2, solver="sparse_cg", random_state=RANDOM_STATE))
    ])


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_confusion_matrix(y_test, pred, target_names, title):
    """Build confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
    ax.xaxis.set_ticklabels(target_names, rotation=45, ha="right")
    ax.yaxis.set_ticklabels(target_names)
    ax.set_title(title)
    plt.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit/predict, confusion matrix
# =========================================================================


def sklearn_way(data):
    """Eager sklearn: fit RidgeClassifier on sparse features, predict, compute
    accuracy and confusion matrix.

    Returns dict with accuracy, predictions, and y_test for plotting.
    """
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    target_names = data["target_names"]

    # Build and fit classifier
    clf = _build_pipeline()
    clf.fit(X_train, y_train)

    # Predict
    pred = clf.predict(X_test)

    # Compute accuracy
    acc = accuracy_score(y_test, pred)
    print(f"sklearn accuracy: {acc:.3f}")

    return {
        "acc": acc,
        "pred": pred,
        "y_test": y_test,
        "target_names": target_names,
    }


# =========================================================================
# XORQ WAY -- deferred fit/predict, deferred metrics and plot
# =========================================================================


def xorq_way(data):
    """Deferred xorq: wrap RidgeClassifier in Pipeline.from_instance, fit/predict
    deferred on registered dense feature tables, compute deferred accuracy.
    Returns ONLY deferred expressions.

    Returns dict with deferred metrics and predictions expressions.
    """
    con = xo.connect()

    # Convert sparse matrices to dense arrays for registration
    # xorq currently works with dense features
    X_train = data["X_train"].toarray()
    X_test = data["X_test"].toarray()
    y_train = data["y_train"]
    y_test = data["y_test"]

    # Create feature column names
    n_features = X_train.shape[1]
    feature_cols = tuple(f"f{i}" for i in range(n_features))

    # Build dataframes
    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df["target"] = y_train

    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df["target"] = y_test

    # Register tables
    train_table = con.register(train_df, "train_newsgroups")
    test_table = con.register(test_df, "test_newsgroups")

    # Wrap sklearn classifier
    sklearn_clf = _build_pipeline()
    xorq_pipe = Pipeline.from_instance(sklearn_clf)

    # Deferred fit/predict
    fitted = xorq_pipe.fit(train_table, features=feature_cols, target="target")
    preds = fitted.predict(test_table, name="pred")

    # Deferred metrics
    make_metric = deferred_sklearn_metric(target="target", pred="pred")
    metrics = preds.agg(acc=make_metric(metric=accuracy_score))

    return {
        "metrics": metrics,
        "preds": preds,
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    print("Loading data...")
    data = _load_data()

    print("\n=== SKLEARN WAY ===")
    sk_results = sklearn_way(data)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(data)

    # Execute deferred metrics
    print("\n=== ASSERTIONS ===")
    xo_metrics_df = xo_results["metrics"].execute()
    xo_acc = xo_metrics_df["acc"].iloc[0]
    print(f"xorq accuracy:   {xo_acc:.3f}")

    # Assert numerical equivalence
    np.testing.assert_allclose(sk_results["acc"], xo_acc, rtol=1e-2)
    print("Assertions passed: sklearn and xorq metrics match.")

    # Build deferred plot for xorq - HAPPENS IN MAIN, NOT IN xorq_way
    print("\n=== GENERATING PLOTS ===")
    target_names = data["target_names"]

    def _build_confusion_matrix_plot(df):
        """Build confusion matrix from materialized predictions."""
        y_true = df["target"].values
        y_pred = df["pred"].values
        return _plot_confusion_matrix(
            y_true, y_pred, target_names, "Confusion Matrix (xorq)"
        )

    xo_png = deferred_matplotlib_plot(xo_results["preds"], _build_confusion_matrix_plot).execute()

    # Build sklearn confusion matrix
    sk_fig = _plot_confusion_matrix(
        sk_results["y_test"],
        sk_results["pred"],
        sk_results["target_names"],
        "Confusion Matrix (sklearn)",
    )

    # Composite: sklearn (left) | xorq (right)
    xo_img = load_plot_bytes(xo_png)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn")
    axes[0].axis("off")

    axes[1].imshow(xo_img)
    axes[1].set_title("xorq")
    axes[1].axis("off")

    plt.suptitle(
        "Document Classification (20 newsgroups): sklearn vs xorq", fontsize=14
    )
    plt.tight_layout()
    out = "imgs/document_classification_20newsgroups.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Composite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
