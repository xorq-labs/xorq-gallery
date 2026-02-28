"""K-Means clustering on handwritten digits
===========================================

sklearn: Load digits dataset, compare KMeans initialization strategies
(k-means++ and random) using StandardScaler preprocessing, evaluate
with homogeneity, completeness, V-measure, ARI, and AMI metrics,
visualize PCA-reduced clusters.

xorq: Same KMeans pipelines wrapped in Pipeline.from_instance, deferred
fit/predict, deferred metrics evaluation, deferred PCA reduction and plotting.
PCA-based initialization is skipped because KMeans(init=pca.components_) passes
a numpy array, which is unhashable for xorq's frozen parameter storage.

Both produce identical clustering metrics (for k-means++ and random init).
Silhouette scores are computed outside the comparator (needs raw data, not
just labels/preds).

Dataset: load_digits (sklearn handwritten digits 0-9)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    split_data_nop,
)
from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
    save_fig,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NS = N_SAMPLES, N_FEATURES, N_DIGITS = (1797, 64, 10)

PRED_COL = "pred"
LABEL_COL = "label"
FEATURE_PREFIX = "f"
FEATURE_COLS = tuple(f"{FEATURE_PREFIX}{i}" for i in range(N_FEATURES))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Load the digits dataset and return as DataFrame."""
    data, labels = load_digits(return_X_y=True)
    (n_samples, n_features), n_digits = data.shape, np.unique(labels).size
    assert NS == (n_samples, n_features, n_digits)
    return pd.DataFrame(data, columns=FEATURE_COLS).assign(**{LABEL_COL: labels})


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_pca_clusters(reduced_data, kmeans_fitted, h=0.02):
    """Build PCA-reduced cluster visualization plot."""
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans_fitted.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    centroids = kmeans_fitted.cluster_centers_
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    ax.set_title(
        "K-means clustering on the digits dataset (PCA-reduced data)\n"
        "Centroids are marked with white cross"
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    fig.tight_layout()
    return fig


def _build_pca_plot(df):
    """Build PCA cluster visualization from a DataFrame."""
    data = df[[col for col in df.columns if col.startswith(FEATURE_PREFIX)]].values
    n_digits = df[LABEL_COL].nunique()
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans_viz = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
    kmeans_viz.fit(reduced_data)
    return _plot_pca_clusters(reduced_data, kmeans_viz)


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def compare_results(comparator):
    assert sorted(sklearn_results := comparator.sklearn_results) == sorted(
        xorq_results := comparator.xorq_results
    )
    metric_names = tuple(name for name, _ in metrics_names_funcs)
    print("\n=== Comparing Results ===")
    for name, sklearn_result in sklearn_results.items():
        xorq_result = xorq_results[name]
        for metric_name in metric_names:
            sk_val = sklearn_result["metrics"][metric_name]
            xo_val = xorq_result["metrics"][metric_name]
            print(
                f"  {name} {metric_name:12s} - sklearn: {sk_val:.3f}, xorq: {xo_val:.3f}"
            )


def plot_results(comparator):
    xo_png = deferred_matplotlib_plot(
        xo.memtable(comparator.df), _build_pca_plot
    ).execute()
    sk_fig = _build_pca_plot(comparator.df)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn")
    axes[0].axis("off")
    axes[1].imshow(load_plot_bytes(xo_png))
    axes[1].set_title("xorq")
    axes[1].axis("off")
    fig.suptitle("K-Means Clustering on Digits: sklearn vs xorq", fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = (KMEANS_PP, RANDOM) = ("k-means++", "random")
names_pipelines = (
    (
        KMEANS_PP,
        SklearnPipeline(
            [
                ("standardscaler", StandardScaler()),
                (
                    "kmeans",
                    KMeans(
                        init="k-means++", n_clusters=N_DIGITS, n_init=4, random_state=0
                    ),
                ),
            ]
        ),
    ),
    (
        RANDOM,
        SklearnPipeline(
            [
                ("standardscaler", StandardScaler()),
                (
                    "kmeans",
                    KMeans(
                        init="random", n_clusters=N_DIGITS, n_init=4, random_state=0
                    ),
                ),
            ]
        ),
    ),
)
metrics_names_funcs = (
    ("homogeneity", metrics.homogeneity_score),
    ("completeness", metrics.completeness_score),
    ("v_measure", metrics.v_measure_score),
    ("ari", metrics.adjusted_rand_score),
    ("ami", metrics.adjusted_mutual_info_score),
)

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=LABEL_COL,
    pred=PRED_COL,
    metrics_names_funcs=metrics_names_funcs,
    load_data=load_data,
    split_data=split_data_nop,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)
# expose the exprs to invoke `xorq build plot_kmeans_digits.py --expr $expr_name`
(xorq_kmeans_pp_preds, xorq_random_preds) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    comparator.result_comparison

    # Silhouette: needs raw data alongside labels_, so computed outside comparator
    data = comparator.df[list(FEATURE_COLS)].values
    print("\n=== Silhouette Scores ===")
    for name, sklearn_result in comparator.sklearn_results.items():
        sil = metrics.silhouette_score(
            data,
            sklearn_result["fitted"].labels_,
            metric="euclidean",
            sample_size=300,
        )
        print(f"  {name}: {sil:.3f}")

    save_fig("imgs/plot_kmeans_digits.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
