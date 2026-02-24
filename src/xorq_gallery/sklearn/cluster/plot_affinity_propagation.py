"""Demo of affinity propagation clustering algorithm
=====================================================

sklearn: Generate synthetic 2D blob data with three centers, fit
AffinityPropagation with preference=-50, evaluate with homogeneity,
completeness, V-measure, ARI, AMI, and silhouette metrics, plot clusters
with lines connecting each point to its cluster center.

xorq: Same AffinityPropagation wrapped in Pipeline.from_instance, deferred
fit/predict, deferred clustering metrics via deferred_sklearn_metric,
deferred cluster plot via deferred_matplotlib_plot.

Both produce identical clustering metrics and visualisations.

Dataset: make_blobs (sklearn synthetic, 300 samples, 3 centers)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline as SklearnPipeline
from toolz import curry
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

RANDOM_STATE = 0
PREFERENCE = -50
FEATURES = ("f0", "f1")
TRUE_LABEL_COL = "true_label"
PRED_COL = "pred"

CLUSTERING_METRICS = (
    ("homogeneity", metrics.homogeneity_score),
    ("completeness", metrics.completeness_score),
    ("v_measure", metrics.v_measure_score),
    ("ari", metrics.adjusted_rand_score),
    ("ami", metrics.adjusted_mutual_info_score),
)


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Generate synthetic 2D blobs with three centers."""
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(
        n_samples=300, centers=centers, cluster_std=0.5, random_state=RANDOM_STATE
    )
    df = pd.DataFrame(X, columns=list(FEATURES))
    df[TRUE_LABEL_COL] = labels_true
    return df


def _build_pipeline():
    """Return sklearn Pipeline wrapping AffinityPropagation."""
    return SklearnPipeline(
        [
            (
                "af",
                AffinityPropagation(preference=PREFERENCE, random_state=RANDOM_STATE),
            ),
        ]
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
            X[class_members, 0],
            X[class_members, 1],
            color=col["color"],
            marker=".",
        )
        ax.scatter(
            cluster_center[0],
            cluster_center[1],
            s=14,
            color=col["color"],
            marker="o",
        )
        for x in X[class_members]:
            ax.plot(
                [cluster_center[0], x[0]],
                [cluster_center[1], x[1]],
                color=col["color"],
            )

    ax.set_title(f"{title_prefix}Estimated number of clusters: {n_clusters}")
    fig.tight_layout()
    return fig


@curry
def _build_cluster_plot(df):
    """Curried plot function for deferred_matplotlib_plot.

    Refits AffinityPropagation on the materialised DataFrame to obtain
    cluster_centers_indices_ (which is not available from predict alone).
    """
    X = df[list(FEATURES)].values

    af = AffinityPropagation(preference=PREFERENCE, random_state=RANDOM_STATE)
    af.fit(X)
    labels = af.labels_
    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters = len(cluster_centers_indices)

    return _plot_clusters(
        X, labels, cluster_centers_indices, n_clusters, title_prefix="xorq: "
    )


# =========================================================================
# SKLEARN WAY -- eager fit, clustering metrics
# =========================================================================


def sklearn_way(df):
    """Eager sklearn: fit AffinityPropagation, compute clustering metrics.

    Returns dict with labels, cluster info, and metric values.
    """
    X = df[list(FEATURES)].values
    labels_true = df[TRUE_LABEL_COL].values

    pipeline = _build_pipeline()
    pipeline.fit(X)

    af = pipeline.named_steps["af"]
    labels = af.labels_
    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters = len(cluster_centers_indices)

    print(f"  Estimated number of clusters: {n_clusters}")

    metric_results = {}
    for name, metric_fn in CLUSTERING_METRICS:
        val = metric_fn(labels_true, labels)
        metric_results[name] = val
        print(f"  {name:15s}: {val:.3f}")

    sil = metrics.silhouette_score(X, labels, metric="sqeuclidean")
    metric_results["silhouette"] = sil
    print(f"  {'silhouette':15s}: {sil:.3f}")

    return {
        "labels": labels,
        "cluster_centers_indices": cluster_centers_indices,
        "n_clusters": n_clusters,
        "metrics": metric_results,
    }


# =========================================================================
# XORQ WAY -- deferred fit/predict, deferred metrics
# =========================================================================


def xorq_way(table):
    """Deferred xorq: Pipeline.from_instance + fit + predict + deferred metrics.

    Returns dict with deferred predictions and metrics expressions.
    No .execute() calls here.
    """
    pipeline = _build_pipeline()
    xorq_pipe = Pipeline.from_instance(pipeline)
    fitted = xorq_pipe.fit(table, features=FEATURES, target=TRUE_LABEL_COL)
    preds = fitted.predict(table, name=PRED_COL)

    make_metric = deferred_sklearn_metric(target=TRUE_LABEL_COL, pred=PRED_COL)
    metrics_expr = preds.agg(
        **{name: make_metric(metric=fn) for name, fn in CLUSTERING_METRICS}
    )

    return {
        "preds": preds,
        "metrics": metrics_expr,
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()

    con = xo.connect()
    table = con.register(df, "blobs")

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(df)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(table)

    # --- Execute deferred metrics and assert equivalence ---
    print("\n=== ASSERTIONS ===")
    xo_metrics_df = xo_results["metrics"].execute()

    sk_metric_vals = {}
    xo_metric_vals = {}
    for name, _ in CLUSTERING_METRICS:
        sk_val = sk_results["metrics"][name]
        xo_val = xo_metrics_df[name].iloc[0]
        print(f"  {name:15s} | sklearn: {sk_val:.3f}  xorq: {xo_val:.3f}")
        sk_metric_vals[name] = [sk_val]
        xo_metric_vals[name] = [xo_val]

    pd.testing.assert_frame_equal(
        pd.DataFrame(sk_metric_vals),
        pd.DataFrame(xo_metric_vals),
        rtol=1e-5,
        check_dtype=False,
    )
    print("Assertions passed: sklearn and xorq clustering metrics match.")

    # --- Plotting ---
    print("\n=== PLOTTING ===")

    X = df[list(FEATURES)].values

    # sklearn plot (eager, in main)
    sk_fig = _plot_clusters(
        X,
        sk_results["labels"],
        sk_results["cluster_centers_indices"],
        sk_results["n_clusters"],
        title_prefix="sklearn: ",
    )

    # xorq deferred plot
    xo_png = deferred_matplotlib_plot(table, _build_cluster_plot).execute()
    xo_img = load_plot_bytes(xo_png)

    # Composite side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(xo_img)
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(
        "Affinity Propagation Clustering: sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    out = "imgs/plot_affinity_propagation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
