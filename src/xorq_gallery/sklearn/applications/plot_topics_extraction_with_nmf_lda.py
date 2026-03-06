"""Topic extraction with NMF and Latent Dirichlet Allocation
=============================================================

sklearn: Vectorize 20 Newsgroups with TfidfVectorizer/CountVectorizer,
fit NMF and LDA decompositions eagerly, display top words per topic.

xorq: Same vectorizer + decomposition pipelines wrapped in
Pipeline.from_instance, fit/transform deferred via SklearnXorqComparator
with the fit_transform variant. Same topic extraction, same results.

Dataset: 20 Newsgroups (2000 documents)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/applications/plot_topics_extraction_with_nmf_lda.py
"""

from __future__ import annotations

from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline as SklearnPipeline

from sklearn.base import clone

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    make_xorq_fit_transform_result,
    split_data_nop,
)
from xorq_gallery.sklearn.sklearn_lib import (
    make_deferred_xorq_fit_transform_result as _make_deferred_xorq_fit_transform_result,
)
from xorq_gallery.utils import fig_to_image, save_fig


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20

TEXT_COL = "text"
LABEL_COL = "label"

tfidf_params = dict(
    max_df=0.95, min_df=2, max_features=n_features, stop_words="english"
)
count_params = dict(
    max_df=0.95, min_df=2, max_features=n_features, stop_words="english"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@cache
def _load_data():
    newsgroups = fetch_20newsgroups(
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    texts = newsgroups.data[:n_samples]
    targets = newsgroups.target[:n_samples]
    return pd.DataFrame({TEXT_COL: texts, LABEL_COL: targets})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_top_words(model, feature_names, n_top=n_top_words):
    """Extract top words per topic from a fitted decomposition model."""
    return [
        list(np.array(feature_names)[topic.argsort()[-n_top:][::-1]])
        for topic in model.components_
    ]


def plot_top_words(model, feature_names, n_top, title=None):
    """Plot top words per topic for a single decomposition model."""
    n_topics = model.components_.shape[0]
    fig, axes = plt.subplots(n_topics, 1, figsize=(7, 2.5 * n_topics))

    for topic_idx in range(n_topics):
        topic = model.components_[topic_idx]
        top_idx = topic.argsort()[-n_top:]
        top_features = feature_names[top_idx]
        weights = topic[top_idx]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontsize=9, pad=8)
        ax.tick_params(axis="both", which="major", labelsize=7)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Pipeline definitions
# ---------------------------------------------------------------------------

methods = (NMF_NAME, LDA_NAME) = ("NMF (Frobenius)", "LDA")

names_pipelines = (
    (
        NMF_NAME,
        SklearnPipeline(
            [
                ("tfidf", TfidfVectorizer(**tfidf_params)),
                (
                    "nmf",
                    NMF(
                        n_components=n_components,
                        random_state=42,
                        beta_loss="frobenius",
                        max_iter=300,
                    ),
                ),
            ]
        ),
    ),
    (
        LDA_NAME,
        SklearnPipeline(
            [
                ("count", CountVectorizer(**count_params)),
                (
                    "lda",
                    LatentDirichletAllocation(
                        n_components=n_components,
                        max_iter=5,
                        learning_method="online",
                        learning_offset=50.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
    ),
)


# ---------------------------------------------------------------------------
# make_other overrides: store vectorizer + decomp model for topic inspection
# ---------------------------------------------------------------------------


def _make_sklearn_other(fitted):
    vectorizer = fitted.steps[0][1]
    decomp = fitted.steps[1][1]
    feature_names = np.array(vectorizer.get_feature_names_out())
    return {
        "decomp": decomp,
        "feature_names": feature_names,
        "topics": get_top_words(decomp, feature_names),
    }


def _make_xorq_other(xorq_fitted):
    vectorizer = xorq_fitted.fitted_steps[0].model
    decomp = xorq_fitted.fitted_steps[1].model
    feature_names = np.array(vectorizer.get_feature_names_out())
    topics = get_top_words(decomp, feature_names)
    return {
        "decomp": lambda: decomp,
        "feature_names": lambda: feature_names,
        "topics": lambda: topics,
    }


def make_sklearn_fit_transform_result(
    pipeline, train_data, test_data, features, target, metrics_names_funcs
):
    """Custom fit_transform for text vectorizers that need raw text strings."""
    X_train = train_data[features[0]].values
    X_test = test_data[features[0]].values
    fitted = clone(pipeline).fit(X_train)
    transformed = fitted.transform(X_test)
    other = _make_sklearn_other(fitted)
    return {
        "fitted": fitted.steps[-1][-1],
        "transformed": transformed,
        "metrics": {},
        "other": other,
    }


make_deferred_xorq_fit_transform_result = _make_deferred_xorq_fit_transform_result(
    make_other=_make_xorq_other
)


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def compare_results(comparator):
    assert sorted(sklearn_results := comparator.sklearn_results) == sorted(
        xorq_results := comparator.xorq_results
    )
    print("\n=== Comparing Results ===")
    for name in sklearn_results:
        sk_topics = sklearn_results[name]["other"]["topics"]
        xo_topics = xorq_results[name]["other"]["topics"]
        print(f"  {name}:")
        for i, (sk_words, xo_words) in enumerate(zip(sk_topics[:3], xo_topics[:3])):
            print(f"    Topic {i + 1} sklearn: {', '.join(sk_words[:8])}")
            print(f"    Topic {i + 1} xorq:    {', '.join(xo_words[:8])}")
        print(f"    ... ({n_components} topics total)")


def plot_results(comparator):
    fig_panels = []
    for name in methods:
        sk_decomp = comparator.sklearn_results[name]["other"]["decomp"]
        sk_fnames = comparator.sklearn_results[name]["other"]["feature_names"]
        xo_decomp = comparator.xorq_results[name]["other"]["decomp"]
        xo_fnames = comparator.xorq_results[name]["other"]["feature_names"]

        sk_fig = plot_top_words(sk_decomp, sk_fnames, n_top_words)
        xo_fig = plot_top_words(xo_decomp, xo_fnames, n_top_words)

        comp_fig, axes = plt.subplots(1, 2, figsize=(14, 2.5 * n_components))
        axes[0].imshow(fig_to_image(sk_fig))
        axes[0].set_title("sklearn")
        axes[0].axis("off")
        axes[1].imshow(fig_to_image(xo_fig))
        axes[1].set_title("xorq")
        axes[1].axis("off")

        comp_fig.suptitle(f"{name}: sklearn vs xorq", fontsize=13)
        comp_fig.tight_layout(rect=[0, 0, 1, 0.99])

        plt.close(sk_fig)
        plt.close(xo_fig)
        fig_panels.append((name, comp_fig))

    return fig_panels


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=(TEXT_COL,),
    target=LABEL_COL,
    pred="pred",
    metrics_names_funcs=(),
    load_data=_load_data,
    split_data=split_data_nop,
    make_sklearn_result=make_sklearn_fit_transform_result,
    make_deferred_xorq_result=make_deferred_xorq_fit_transform_result,
    make_xorq_result=make_xorq_fit_transform_result,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)

# Module-level deferred exprs
(xorq_nmf_transformed, xorq_lda_transformed) = (
    comparator.deferred_xorq_results[name]["transformed"] for name in methods
)


def main():
    comparator.result_comparison
    fig_panels = comparator.plot_results()
    for name, fig in fig_panels:
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_fig(f"imgs/topics_{fname}.png", fig)


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
