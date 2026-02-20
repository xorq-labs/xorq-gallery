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

from xorq_gallery.utils import deferred_matplotlib_plot, fig_to_image, load_plot_bytes


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20

tfidf_params = dict(
    max_df=0.95, min_df=2, max_features=n_features, stop_words="english"
)
count_params = dict(
    max_df=0.95, min_df=2, max_features=n_features, stop_words="english"
)


def _load_data():
    newsgroups = fetch_20newsgroups(
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    texts = newsgroups.data[:n_samples]
    targets = newsgroups.target[:n_samples]
    return texts, targets


def get_top_words(model, feature_names, n_top=n_top_words):
    """Extract top words per topic from a fitted decomposition model."""
    topics = []
    for topic in model.components_:
        top_indices = topic.argsort()[-n_top:][::-1]
        topics.append(list(feature_names[top_indices]))
    return topics


def plot_top_words(model, feature_names, n_top, title):
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
        ax.set_title(f"Topic {topic_idx + 1}", fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=7)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


def _build_topics_figure(model, feature_names, pipe_name):
    """Return a UDAF-compatible plotting function for topic extraction results."""
    def _plot(_df):
        return plot_top_words(model, feature_names, n_top_words, f"xorq - {pipe_name}")
    return _plot


# ---------------------------------------------------------------------------
# Pipeline definitions (shared sklearn Pipeline objects)
# ---------------------------------------------------------------------------


def _build_pipelines():
    return {
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


def sklearn_way(texts, pipelines):
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


def xorq_way(texts, targets, pipelines):
    """Deferred xorq: register text as ibis, Pipeline.from_instance.

    Returns a dict mapping pipeline name to a deferred plot expression
    (PNG bytes via UDAF).  The fit goes through ibis data, then fitted
    model components are captured in the closure for visualization.
    """
    con = xo.connect()
    df = pd.DataFrame({"text": texts, "label": targets})
    data = con.register(df, "newsgroups")

    plot_exprs = {}
    for name, sk_pipe in pipelines.items():
        xorq_pipe = Pipeline.from_instance(sk_pipe)
        fitted = xorq_pipe.fit(data, features=("text",), target="label")

        # Access fitted sklearn objects for topic inspection
        vectorizer = fitted.fitted_steps[0].model
        decomp = fitted.fitted_steps[1].model
        feature_names = np.array(vectorizer.get_feature_names_out())
        topics = get_top_words(decomp, feature_names)

        print(f"  xorq   {name}:")
        for i, words in enumerate(topics[:3]):
            print(f"    Topic {i + 1}: {', '.join(words[:8])}")
        print(f"    ... ({n_components} topics total)")

        plot_exprs[name] = deferred_matplotlib_plot(
            data, _build_topics_figure(decomp, feature_names, name)
        )

    return plot_exprs


# =========================================================================
# Run and plot side by side
# =========================================================================

def main():
    os.makedirs("imgs", exist_ok=True)

    texts, targets = _load_data()
    pipelines = _build_pipelines()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(texts, pipelines)

    print("\n=== XORQ WAY ===")
    plot_exprs = xorq_way(texts, targets, pipelines)

    # For each method: build sklearn figure natively, execute xorq deferred,
    # composite side by side
    for name in pipelines:
        # Execute xorq deferred plot
        xo_png = plot_exprs[name].execute()

        # Build sklearn figure natively
        sk_model = sk_results[name]["model"]
        sk_feature_names = sk_results[name]["feature_names"]
        sk_fig = plot_top_words(
            sk_model, sk_feature_names, n_top_words, f"sklearn - {name}"
        )

        # Composite: sklearn (left) | xorq deferred (right)
        xo_img = load_plot_bytes(xo_png)

        fig, axes = plt.subplots(1, 2, figsize=(14, 2.5 * n_components))
        axes[0].imshow(fig_to_image(sk_fig))
        axes[0].set_title("sklearn")
        axes[0].axis("off")
        axes[1].imshow(xo_img)
        axes[1].set_title("xorq")
        axes[1].axis("off")

        plt.suptitle(f"{name}: sklearn vs xorq", fontsize=13)
        plt.tight_layout()

        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out = f"imgs/topics_{fname}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
