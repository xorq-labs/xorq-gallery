"""Selecting the number of clusters with silhouette analysis on KMeans clustering
================================================================================

sklearn: Generate synthetic clustered data using make_blobs, iterate over
candidate cluster counts (2-6), fit KMeans for each, compute silhouette scores
per sample and overall average, visualize silhouette plots and cluster scatter
plots to identify optimal cluster count.

xorq: Same KMeans models wrapped in Pipeline.from_instance, fit/predict
deferred for multiple cluster counts, silhouette scores computed outside
comparator (require raw feature data alongside cluster labels).

Both produce identical silhouette scores and cluster assignments.

Dataset: make_blobs (synthetic 2D clustered data with 500 samples, 4 centers)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/cluster/plot_kmeans_silhouette_analysis.py
"""

from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline as SklearnPipeline

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    split_data_nop,
)
from xorq_gallery.utils import (
    fig_to_image,
    save_fig,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANGE_N_CLUSTERS = (2, 3, 4, 5, 6)
FEATURE_COLS = ("f0", "f1")
TARGET_COL = "true_label"
PRED_COL = "pred"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Generate synthetic clustered data using make_blobs."""
    X, y = make_blobs(
        n_samples=500,
        n_features=2,
        centers=4,
        cluster_std=1,
        center_box=(-10.0, 10.0),
        shuffle=True,
        random_state=1,
    )
    return pd.DataFrame(X, columns=list(FEATURE_COLS)).assign(**{TARGET_COL: y})


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_silhouette_analysis(
    X, n_clusters, cluster_labels, silhouette_avg, sample_silhouette_values, centers
):
    """Build silhouette analysis plot (silhouette plot + cluster scatter)."""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_values = sample_silhouette_values[cluster_labels == i]
        ith_values.sort()
        size_cluster_i = ith_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    fig.suptitle(
        f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {n_clusters}",
        fontsize=14,
        fontweight="bold",
    )
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
        sk_h = sklearn_result["metrics"]["homogeneity"]
        xo_h = xorq_result["metrics"]["homogeneity"]
        print(
            f"  n_clusters={name:2s} homogeneity - sklearn: {sk_h:.4f}, xorq: {xo_h:.4f}"
        )

    # Silhouette: needs raw feature data, computed outside comparator
    X = comparator.df[list(FEATURE_COLS)].values
    print("\n=== Silhouette Scores ===")
    for name in methods:
        n_clusters = int(name)
        sk_labels = sklearn_results[name]["preds"]
        xo_labels = xorq_results[name]["preds"][PRED_COL].values
        sk_sil = sk_metrics.silhouette_score(X, sk_labels)
        xo_sil = sk_metrics.silhouette_score(X, xo_labels)
        print(f"  n_clusters={n_clusters}: sklearn={sk_sil:.3f}, xorq={xo_sil:.3f}")
        np.testing.assert_allclose(sk_sil, xo_sil, rtol=1e-5)
    print("Silhouette scores match.")


def plot_results(comparator):
    X = comparator.df[list(FEATURE_COLS)].values
    row_figs = []

    for name in methods:
        n_clusters = int(name)
        km = comparator.sklearn_results[name]["fitted"]
        sk_labels = comparator.sklearn_results[name]["preds"]
        xo_labels = comparator.xorq_results[name]["preds"][PRED_COL].values
        centers = km.cluster_centers_

        sk_sil_avg = sk_metrics.silhouette_score(X, sk_labels)
        sk_sil_samples = sk_metrics.silhouette_samples(X, sk_labels)
        xo_sil_avg = sk_metrics.silhouette_score(X, xo_labels)
        xo_sil_samples = sk_metrics.silhouette_samples(X, xo_labels)

        sk_fig = _plot_silhouette_analysis(
            X, n_clusters, sk_labels, sk_sil_avg, sk_sil_samples, centers
        )
        xo_fig = _plot_silhouette_analysis(
            X, n_clusters, xo_labels, xo_sil_avg, xo_sil_samples, centers
        )

        row_fig, row_axes = plt.subplots(1, 2, figsize=(36, 7))
        row_axes[0].imshow(fig_to_image(sk_fig))
        row_axes[0].set_title("sklearn", fontsize=14)
        row_axes[0].axis("off")
        row_axes[1].imshow(fig_to_image(xo_fig))
        row_axes[1].set_title("xorq", fontsize=14)
        row_axes[1].axis("off")
        row_fig.suptitle(f"n_clusters={n_clusters}", fontsize=14)
        row_fig.tight_layout()
        row_figs.append(row_fig)
        plt.close(sk_fig)
        plt.close(xo_fig)

    fig, axes = plt.subplots(len(methods), 1, figsize=(36, 7 * len(methods)))
    for row, rf in enumerate(row_figs):
        axes[row].imshow(fig_to_image(rf))
        axes[row].axis("off")
        plt.close(rf)
    fig.suptitle("Silhouette Analysis: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = tuple(str(n) for n in RANGE_N_CLUSTERS)
names_pipelines = tuple(
    (str(n), SklearnPipeline([("kmeans", KMeans(n_clusters=n, random_state=10))]))
    for n in RANGE_N_CLUSTERS
)
metrics_names_funcs = (("homogeneity", sk_metrics.homogeneity_score),)

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=TARGET_COL,
    pred=PRED_COL,
    metrics_names_funcs=metrics_names_funcs,
    load_data=load_data,
    split_data=split_data_nop,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)
# expose the exprs to invoke `xorq build plot_kmeans_silhouette_analysis.py --expr $expr_name`
(
    xorq_k2_preds,
    xorq_k3_preds,
    xorq_k4_preds,
    xorq_k5_preds,
    xorq_k6_preds,
) = (comparator.deferred_xorq_results[name]["preds"] for name in methods)


def main():
    comparator.result_comparison
    save_fig("imgs/plot_kmeans_silhouette_analysis.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
