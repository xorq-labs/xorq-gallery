"""Topic extraction with NMF and Latent Dirichlet Allocation
=============================================================

sklearn: Vectorize 20 Newsgroups with TfidfVectorizer/CountVectorizer,
fit NMF and LDA decompositions eagerly, display top words per topic.

xorq: Register text corpus as an ibis expression, build vectorizer +
decomposition as a single Pipeline.from_instance, fit deferred.
Same topic extraction, same results.

Dataset: 20 Newsgroups (2000 documents)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline as SklearnPipeline
from xorq.expr.ml.pipeline_lib import Pipeline


# ---------------------------------------------------------------------------
# Shared config and data
# ---------------------------------------------------------------------------

n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20

newsgroups = fetch_20newsgroups(
    remove=("headers", "footers", "quotes"),
    shuffle=True,
    random_state=42,
)
texts = newsgroups.data[:n_samples]
targets = newsgroups.target[:n_samples]

tfidf_params = dict(
    max_df=0.95, min_df=2, max_features=n_features, stop_words="english"
)
count_params = dict(
    max_df=0.95, min_df=2, max_features=n_features, stop_words="english"
)


def get_top_words(model, feature_names, n_top=n_top_words):
    """Extract top words per topic from a fitted decomposition model."""
    topics = []
    for topic in model.components_:
        top_indices = topic.argsort()[-n_top:][::-1]
        topics.append(list(feature_names[top_indices]))
    return topics


def plot_top_words_side_by_side(sk_model, xo_model, feature_names, n_top, title):
    """Plot top words for sklearn (left) vs xorq (right)."""
    n_topics = sk_model.components_.shape[0]
    fig, axes = plt.subplots(n_topics, 2, figsize=(14, 2.5 * n_topics))

    for topic_idx in range(n_topics):
        for col, (label, model) in enumerate(
            [("sklearn", sk_model), ("xorq", xo_model)]
        ):
            topic = model.components_[topic_idx]
            top_idx = topic.argsort()[-n_top:]
            top_features = feature_names[top_idx]
            weights = topic[top_idx]

            ax = axes[topic_idx, col]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f"{label} Topic {topic_idx + 1}", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Pipeline definitions (shared sklearn Pipeline objects)
# ---------------------------------------------------------------------------

pipelines = {
    "NMF (Frobenius)": SklearnPipeline(
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
    "LDA": SklearnPipeline(
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
}


# =========================================================================
# SKLEARN WAY
# =========================================================================


def sklearn_way(texts):
    """Eager sklearn: vectorize + decompose on raw text list."""
    results = {}
    for name, pipe in pipelines.items():
        pipe.fit(texts)
        vectorizer = pipe.steps[0][1]
        decomp = pipe.steps[1][1]
        feature_names = np.array(vectorizer.get_feature_names_out())
        topics = get_top_words(decomp, feature_names)
        results[name] = {
            "model": decomp,
            "feature_names": feature_names,
            "topics": topics,
        }
        print(f"  sklearn {name}:")
        for i, words in enumerate(topics[:3]):
            print(f"    Topic {i + 1}: {', '.join(words[:8])}")
        print(f"    ... ({n_components} topics total)")
    return results


# =========================================================================
# XORQ WAY
# =========================================================================


def xorq_way(texts, targets):
    """Deferred xorq: register text as ibis, Pipeline.from_instance."""
    con = xo.connect()
    df = pd.DataFrame({"text": texts, "label": targets})
    data = con.register(df, "newsgroups")

    results = {}
    for name, sk_pipe in pipelines.items():
        xorq_pipe = Pipeline.from_instance(sk_pipe)
        fitted = xorq_pipe.fit(data, features=("text",), target="label")

        # Access fitted sklearn objects for topic inspection
        vectorizer = fitted.fitted_steps[0].model
        decomp = fitted.fitted_steps[1].model
        feature_names = np.array(vectorizer.get_feature_names_out())
        topics = get_top_words(decomp, feature_names)

        results[name] = {
            "model": decomp,
            "feature_names": feature_names,
            "topics": topics,
        }
        print(f"  xorq   {name}:")
        for i, words in enumerate(topics[:3]):
            print(f"    Topic {i + 1}: {', '.join(words[:8])}")
        print(f"    ... ({n_components} topics total)")
    return results


# =========================================================================
# Run and plot side by side
# =========================================================================

if __name__ in ("__main__", "__pytest_main__"):
    os.makedirs("imgs", exist_ok=True)

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(texts)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(texts, targets)

    # Side-by-side topic comparison plots
    for name in pipelines:
        fig = plot_top_words_side_by_side(
            sk_results[name]["model"],
            xo_results[name]["model"],
            sk_results[name]["feature_names"],
            n_top_words,
            f"{name}: sklearn (left) vs xorq (right)",
        )
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(f"imgs/topics_{fname}.png", dpi=150)

    plt.close("all")

    pytest_examples_passed = True
