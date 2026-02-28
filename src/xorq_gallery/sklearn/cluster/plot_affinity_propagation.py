"""Demo of affinity propagation clustering algorithm
=====================================================

sklearn: Generate synthetic 2D blob data with three centers, fit
AffinityPropagation with preference=-50, evaluate with homogeneity,
completeness, V-measure, ARI, AMI, and silhouette metrics, plot clusters
with lines connecting each point to its cluster center.

xorq: Same AffinityPropagation wrapped in Pipeline.from_instance, deferred
fit/predict, deferred clustering metrics via deferred_sklearn_metric,
deferred cluster plot via deferred_matplotlib_plot.

Both produce identical clustering metrics.

Dataset: make_blobs (sklearn synthetic, 300 samples, 3 centers)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline as SklearnPipeline

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

RANDOM_STATE = 0
PREFERENCE = -50
FEATURE_COLS = ("f0", "f1")
TARGET_COL = "true_label"
PRED_COL = "pred"

CLUSTERING_METRICS = (
    ("homogeneity", metrics.homogeneity_score),
    ("completeness", metrics.completeness_score),
    ("v_measure", metrics.v_measure_score),
    ("ari", metrics.adjusted_rand_score),
    ("ami", metrics.adjusted_mutual_info_score),
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Generate synthetic 2D blobs with three centers."""
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(
        n_samples=300, centers=centers, cluster_std=0.5, random_state=RANDOM_STATE
    )
    return pd.DataFrame(X, columns=list(FEATURE_COLS)).assign(
        **{TARGET_COL: labels_true}
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_clusters(X, labels, cluster_centers_indices, n_clusters, title_prefix=""):
    """Plot clusters with lines connecting each point to its cluster center."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, max(n_clusters, 1))))
    for k, col in zip(range(n_clusters), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        ax.scatter(
            X[class_members, 0], X[class_members, 1], color=col["color"], marker="."
        )
        ax.scatter(
            cluster_center[0], cluster_center[1], s=14, color=col["color"], marker="o"
        )
        for x in X[class_members]:
            ax.plot(
                [cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col["color"]
            )
    ax.set_title(f"{title_prefix}Estimated number of clusters: {n_clusters}")
    fig.tight_layout()
    return fig


def _build_cluster_plot(df):
    """Build cluster plot by refitting AffinityPropagation on materialised DataFrame."""
    X = df[list(FEATURE_COLS)].values
    af = AffinityPropagation(preference=PREFERENCE, random_state=RANDOM_STATE)
    af.fit(X)
    labels = af.labels_
    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters = len(cluster_centers_indices)
    return _plot_clusters(
        X, labels, cluster_centers_indices, n_clusters, title_prefix="xorq: "
    )


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
        for metric_name, _ in CLUSTERING_METRICS:
            sk_val = sklearn_result["metrics"][metric_name]
            xo_val = xorq_result["metrics"][metric_name]
            print(
                f"  {name} {metric_name:15s} - sklearn: {sk_val:.3f}, xorq: {xo_val:.3f}"
            )


def plot_results(comparator):
    af_fitted = comparator.sklearn_results[AP_NAME]["fitted"]
    X = comparator.df[list(FEATURE_COLS)].values
    labels = af_fitted.labels_
    cluster_centers_indices = af_fitted.cluster_centers_indices_
    n_clusters = len(cluster_centers_indices)

    sk_fig = _plot_clusters(
        X, labels, cluster_centers_indices, n_clusters, title_prefix="sklearn: "
    )
    xo_png = deferred_matplotlib_plot(
        xo.memtable(comparator.df), _build_cluster_plot
    ).execute()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(load_plot_bytes(xo_png))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    fig.suptitle("Affinity Propagation Clustering: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = (AP_NAME,) = ("AffinityPropagation",)
names_pipelines = (
    (
        AP_NAME,
        SklearnPipeline(
            [
                (
                    "af",
                    AffinityPropagation(
                        preference=PREFERENCE, random_state=RANDOM_STATE
                    ),
                )
            ]
        ),
    ),
)
metrics_names_funcs = CLUSTERING_METRICS

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
# expose the exprs to invoke `xorq build plot_affinity_propagation.py --expr $expr_name`
(xorq_ap_preds,) = (comparator.deferred_xorq_results[name]["preds"] for name in methods)


def main():
    comparator.result_comparison

    # Silhouette: needs raw data alongside labels_, so computed outside comparator
    af_fitted = comparator.sklearn_results[AP_NAME]["fitted"]
    X = comparator.df[list(FEATURE_COLS)].values
    sil = metrics.silhouette_score(X, af_fitted.labels_, metric="sqeuclidean")
    print(f"\nSilhouette: {sil:.3f}")

    save_fig("imgs/plot_affinity_propagation.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
