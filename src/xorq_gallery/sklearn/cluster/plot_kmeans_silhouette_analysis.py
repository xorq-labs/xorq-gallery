"""Selecting the number of clusters with silhouette analysis on KMeans clustering
================================================================================

sklearn: Generate synthetic clustered data using make_blobs, iterate over
candidate cluster counts (2-6), fit KMeans for each, compute silhouette scores
per sample and overall average, visualize silhouette plots and cluster scatter
plots to identify optimal cluster count.

xorq: Same KMeans models wrapped in Pipeline.from_instance, deferred
fit/predict for multiple cluster counts, deferred silhouette score computation,
deferred plotting of silhouette analysis and cluster visualization.

Both produce identical silhouette scores and visualizations.

Dataset: make_blobs (synthetic 2D clustered data with 500 samples, 4 centers)
"""

from __future__ import annotations

import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.pipeline import make_pipeline
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Generate synthetic clustered data using make_blobs.

    Returns pandas DataFrame with features f0, f1 and true labels.
    """
    X, y = make_blobs(
        n_samples=500,
        n_features=2,
        centers=4,
        cluster_std=1,
        center_box=(-10.0, 10.0),
        shuffle=True,
        random_state=1,
    )
    df = pd.DataFrame(X, columns=["f0", "f1"])
    df["true_label"] = y
    return df


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANGE_N_CLUSTERS = [2, 3, 4, 5, 6]
FEATURES = ("f0", "f1")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_silhouette_analysis(X, n_clusters, cluster_labels, silhouette_avg, sample_silhouette_values, centers):
    """Build silhouette analysis plot (silhouette plot + cluster scatter).

    Parameters
    ----------
    X : ndarray
        Data array (n_samples, 2)
    n_clusters : int
        Number of clusters
    cluster_labels : ndarray
        Cluster assignments
    silhouette_avg : float
        Average silhouette score
    sample_silhouette_values : ndarray
        Silhouette score per sample
    centers : ndarray
        Cluster centers (n_clusters, 2)

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # Silhouette plot
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
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

    # Cluster scatter plot
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Draw cluster centers
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

    plt.suptitle(
        f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {n_clusters}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit/predict, silhouette analysis
# =========================================================================


def sklearn_way(df, n_clusters):
    """Eager sklearn: fit KMeans for a single n_clusters, compute
    silhouette scores, store results for plotting.

    Returns dict with results for this n_clusters value.
    """
    X = df[["f0", "f1"]].values

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    centers = clusterer.cluster_centers_

    return {
        "cluster_labels": cluster_labels,
        "silhouette_avg": silhouette_avg,
        "sample_silhouette_values": sample_silhouette_values,
        "centers": centers,
    }


# =========================================================================
# XORQ WAY -- deferred fit/predict, deferred silhouette analysis
# =========================================================================


def xorq_way(df, n_clusters):
    """Deferred xorq: wrap KMeans in Pipeline.from_instance, fit/predict
    deferred for a single n_clusters value.

    Returns deferred prediction expression.
    Nothing is executed until ``.execute()``.
    """
    con = xo.connect()
    table = con.register(df, f"blobs_n{n_clusters}")

    # Wrap KMeans in sklearn pipeline, then wrap in xorq Pipeline
    sklearn_pipe = make_pipeline(KMeans(n_clusters=n_clusters, random_state=10))
    xorq_pipe = Pipeline.from_instance(sklearn_pipe)

    # Deferred fit and predict
    fitted = xorq_pipe.fit(table, features=FEATURES, target="true_label")
    preds = fitted.predict(table, name="pred")

    return preds


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    print("=== GENERATING DATA ===")
    df = _load_data()
    print(f"Generated {len(df)} samples with 2 features")
    X = df[["f0", "f1"]].values

    print("\n=== SKLEARN WAY ===")
    sk_results = {}
    for n_clusters in RANGE_N_CLUSTERS:
        sk_res = sklearn_way(df, n_clusters)
        sk_results[n_clusters] = sk_res
        print(f"For n_clusters = {n_clusters}, the average silhouette_score is: {sk_res['silhouette_avg']:.3f}")

    print("\n=== XORQ WAY ===")
    xo_preds = {}
    for n_clusters in RANGE_N_CLUSTERS:
        xo_preds[n_clusters] = xorq_way(df, n_clusters)

    # Execute deferred predictions and compute silhouette scores
    print("\n=== COMPUTING XORQ SILHOUETTE SCORES ===")
    xo_silhouette_scores = {}

    for n_clusters in RANGE_N_CLUSTERS:
        preds_df = xo_preds[n_clusters].execute()
        cluster_labels = preds_df["pred"].values

        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        print(f"For n_clusters = {n_clusters}, the average silhouette_score is: {silhouette_avg:.3f}")

        xo_silhouette_scores[n_clusters] = {
            "silhouette_avg": silhouette_avg,
            "sample_silhouette_values": sample_silhouette_values,
            "cluster_labels": cluster_labels,
        }

    # Assert numerical equivalence BEFORE plotting
    print("\n=== ASSERTIONS ===")
    print("Comparing silhouette scores (sklearn vs xorq):")
    for n_clusters in RANGE_N_CLUSTERS:
        sk_avg = sk_results[n_clusters]["silhouette_avg"]
        xo_avg = xo_silhouette_scores[n_clusters]["silhouette_avg"]
        print(f"  n_clusters={n_clusters}: sklearn={sk_avg:.3f}, xorq={xo_avg:.3f}")
        np.testing.assert_allclose(sk_avg, xo_avg, rtol=1e-5)

    print("Assertions passed: sklearn and xorq silhouette scores match.")

    # Create composite plots for each n_clusters value
    print("\n=== PLOTTING ===")
    for n_clusters in RANGE_N_CLUSTERS:
        # Sklearn plot
        sk_res = sk_results[n_clusters]
        sk_fig = _plot_silhouette_analysis(
            X,
            n_clusters,
            sk_res["cluster_labels"],
            sk_res["silhouette_avg"],
            sk_res["sample_silhouette_values"],
            sk_res["centers"],
        )

        # Xorq plot - refit to get centers (matching sklearn approach)
        kmeans_viz = KMeans(n_clusters=n_clusters, random_state=10)
        kmeans_viz.fit(X)
        centers = kmeans_viz.cluster_centers_

        xo_fig = _plot_silhouette_analysis(
            X,
            n_clusters,
            xo_silhouette_scores[n_clusters]["cluster_labels"],
            xo_silhouette_scores[n_clusters]["silhouette_avg"],
            xo_silhouette_scores[n_clusters]["sample_silhouette_values"],
            centers,
        )

        # Composite: sklearn (left) | xorq (right)
        sk_img = fig_to_image(sk_fig)
        xo_img = fig_to_image(xo_fig)

        fig, axes = plt.subplots(1, 2, figsize=(36, 7))
        axes[0].imshow(sk_img)
        axes[0].set_title("sklearn", fontsize=16)
        axes[0].axis("off")

        axes[1].imshow(xo_img)
        axes[1].set_title("xorq", fontsize=16)
        axes[1].axis("off")

        plt.suptitle(
            f"Silhouette Analysis: sklearn vs xorq (n_clusters={n_clusters})",
            fontsize=18,
            fontweight="bold",
        )
        plt.tight_layout()
        out = f"imgs/plot_kmeans_silhouette_analysis_n{n_clusters}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close(sk_fig)
        plt.close(xo_fig)
        print(f"Saved: {out}")

    print(f"\nAll composite plots saved to imgs/")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
