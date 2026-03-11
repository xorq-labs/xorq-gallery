"""Classification of text documents using sparse features
===========================================================

sklearn: Load 20 newsgroups dataset (4 categories), vectorize with TfidfVectorizer
(max_df=0.5, min_df=5, sublinear_tf=True), train RidgeClassifier on training data,
predict on test data, compute accuracy and confusion matrix.

xorq: Same preprocessing pipeline wrapped in Pipeline.from_instance, deferred
fit/predict on registered tables, deferred accuracy metric, deferred confusion matrix plot.

Both produce identical accuracy and confusion matrices.

Dataset: 20 newsgroups (4 categories: alt.atheism, talk.religion.misc, comp.graphics, sci.space)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/text/plot_document_classification_20newsgroups.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.pipeline import Pipeline as SklearnPipeline

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
)
from xorq_gallery.utils import fig_to_image, save_fig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = (
    "alt.atheism",
    "talk.religion.misc",
    "comp.graphics",
    "sci.space",
)

RANDOM_STATE = 42
TARGET_COL = "target"
PRED_COL = "pred"


# ---------------------------------------------------------------------------
# Data loading — pre-vectorize to dense features
# ---------------------------------------------------------------------------


def _load_data():
    """Load and vectorize the 20 newsgroups dataset.

    Returns a single DataFrame with dense TF-IDF features + target column.
    The vectorizer is fit on training data and applied to both train and test
    to produce a single pre-vectorized table.  This lets both sklearn and xorq
    paths operate on the same dense features.
    """
    data_train = fetch_20newsgroups(
        subset="train",
        categories=CATEGORIES,
        shuffle=True,
        random_state=RANDOM_STATE,
        remove=("headers", "footers", "quotes"),
    )
    data_test = fetch_20newsgroups(
        subset="test",
        categories=CATEGORIES,
        shuffle=True,
        random_state=RANDOM_STATE,
        remove=("headers", "footers", "quotes"),
    )

    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
    )
    X_train = vectorizer.fit_transform(data_train.data).toarray()
    X_test = vectorizer.transform(data_test.data).toarray()

    n_features = X_train.shape[1]
    feature_cols = tuple(f"f{i}" for i in range(n_features))

    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df[TARGET_COL] = data_train.target
    train_df["_split"] = "train"

    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df[TARGET_COL] = data_test.target
    test_df["_split"] = "test"

    # Store feature_cols and target_names as module-level state for plotting
    _load_data.feature_cols = feature_cols
    _load_data.target_names = data_train.target_names

    # Combine — split_data will separate them
    return pd.concat([train_df, test_df], ignore_index=True)


def split_data(df):
    """Split by pre-assigned _split column, then drop the marker."""
    train_df = (
        df[df["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
    )
    test_df = df[df["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

names_pipelines = (
    (
        "RidgeClassifier",
        SklearnPipeline(
            [
                (
                    "clf",
                    RidgeClassifier(
                        tol=1e-2, solver="sparse_cg", random_state=RANDOM_STATE
                    ),
                )
            ]
        ),
    ),
)

methods = ("RidgeClassifier",)
metrics_names_funcs = (("accuracy", accuracy_score),)


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def _plot_confusion_matrix(y_test, pred, target_names, title):
    """Build confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
    ax.xaxis.set_ticklabels(target_names, rotation=45, ha="right")
    ax.yaxis.set_ticklabels(target_names)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def compare_results(comparator):
    assert sorted(sklearn_results := comparator.sklearn_results) == sorted(
        xorq_results := comparator.xorq_results
    )
    for name in sklearn_results:
        sk_acc = sklearn_results[name]["metrics"]["accuracy"]
        xo_acc = xorq_results[name]["metrics"]["accuracy"]
        print(f"  {name}: sklearn acc={sk_acc:.3f}, xorq acc={xo_acc:.3f}")
        np.testing.assert_allclose(sk_acc, xo_acc, rtol=1e-2)
    print("Assertions passed.")


def plot_results(comparator):
    target_names = _load_data.target_names
    _, test_df = split_data(comparator.df)

    sk_preds = comparator.sklearn_results["RidgeClassifier"]["preds"]
    xo_preds = comparator.xorq_results["RidgeClassifier"]["preds"][PRED_COL].values
    y_test = test_df[TARGET_COL].values

    sk_fig = _plot_confusion_matrix(
        y_test, sk_preds, target_names, "Confusion Matrix (sklearn)"
    )
    xo_fig = _plot_confusion_matrix(
        y_test, xo_preds, target_names, "Confusion Matrix (xorq)"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn")
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq")
    axes[1].axis("off")

    fig.suptitle(
        "Document Classification (20 newsgroups): sklearn vs xorq", fontsize=14
    )
    fig.tight_layout()

    plt.close(sk_fig)
    plt.close(xo_fig)
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

# Force data load so feature_cols is available for comparator construction
_df = _load_data()
FEATURE_COLS = _load_data.feature_cols

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=TARGET_COL,
    pred=PRED_COL,
    metrics_names_funcs=metrics_names_funcs,
    load_data=lambda: _df,
    split_data=split_data,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)

# Module-level deferred exprs
(xorq_ridge_preds,) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    comparator.result_comparison
    save_fig("imgs/document_classification_20newsgroups.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
