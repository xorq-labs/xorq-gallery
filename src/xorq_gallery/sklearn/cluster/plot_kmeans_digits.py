"""K-Means clustering on handwritten digits
===========================================

sklearn: Load digits dataset, compare three KMeans initialization strategies
(k-means++, random, PCA-based) using StandardScaler preprocessing, evaluate
with homogeneity, completeness, V-measure, ARI, AMI, and silhouette metrics,
visualize PCA-reduced clusters.

xorq: Same KMeans pipelines wrapped in Pipeline.from_instance, deferred
fit/predict, deferred metrics evaluation, deferred PCA reduction and plotting.

Both produce identical clustering metrics.

Dataset: load_digits (sklearn handwritten digits 0-9)
"""

from __future__ import annotations

import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import xorq.api as xo
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
    """Load the digits dataset and return data, labels, metadata."""
    data, labels = load_digits(return_X_y=True)
    (n_samples, n_features), n_digits = data.shape, np.unique(labels).size
    return {
        "data": data,
        "labels": labels,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_digits": n_digits,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_pca_clusters(reduced_data, kmeans_fitted, n_digits, h=0.02):
    """Build PCA-reduced cluster visualization plot.

    Parameters
    ----------
    reduced_data : ndarray
        2D PCA-reduced data
    kmeans_fitted : fitted KMeans instance
        Fitted KMeans model
    n_digits : int
        Number of unique digits (clusters)
    h : float
        Mesh step size

    Returns
    -------
    matplotlib.figure.Figure
    """
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh
    Z = kmeans_fitted.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

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

    # Plot centroids as white X
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

    plt.tight_layout()
    return fig


def _build_pca_plot(df):
    """Build PCA cluster visualization from deferred execution.

    This function receives the full dataset, refits KMeans on PCA-reduced
    data (similar to sklearn approach), and creates the visualization.
    """
    # Extract data from dataframe
    data_cols = [col for col in df.columns if col.startswith("f")]
    data = df[data_cols].values
    n_digits = df["label"].nunique()

    # Reduce to 2D via PCA
    reduced_data = PCA(n_components=2).fit_transform(data)

    # Refit KMeans on reduced data for visualization
    kmeans_viz = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
    kmeans_viz.fit(reduced_data)

    return _plot_pca_clusters(reduced_data, kmeans_viz, n_digits)


# =========================================================================
# SKLEARN WAY -- eager fit/predict, benchmarking
# =========================================================================


def sklearn_way(dataset):
    """Eager sklearn: benchmark three KMeans initialization strategies,
    compute clustering metrics, create PCA visualization."""
    data = dataset["data"]
    labels = dataset["labels"]
    n_digits = dataset["n_digits"]

    def bench_k_means(kmeans, name, data, labels):
        """Benchmark KMeans initialization methods."""
        t0 = time()
        estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
        fit_time = time() - t0
        results = [name, fit_time, estimator[-1].inertia_]

        # Clustering metrics
        clustering_metrics = [
            metrics.homogeneity_score,
            metrics.completeness_score,
            metrics.v_measure_score,
            metrics.adjusted_rand_score,
            metrics.adjusted_mutual_info_score,
        ]
        results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

        # Silhouette score
        results += [
            metrics.silhouette_score(
                data,
                estimator[-1].labels_,
                metric="euclidean",
                sample_size=300,
            )
        ]

        # Format: name, time, inertia, homo, compl, v-meas, ARI, AMI, silhouette
        formatter_result = (
            "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
        )
        print(formatter_result.format(*results))

        return {
            "name": name,
            "fit_time": fit_time,
            "inertia": estimator[-1].inertia_,
            "homogeneity": results[3],
            "completeness": results[4],
            "v_measure": results[5],
            "ari": results[6],
            "ami": results[7],
            "silhouette": results[8],
            "estimator": estimator,
        }

    print(82 * "_")
    print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

    # k-means++ initialization
    kmeans_kpp = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
    res_kpp = bench_k_means(kmeans=kmeans_kpp, name="k-means++", data=data, labels=labels)

    # Random initialization
    kmeans_rand = KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)
    res_rand = bench_k_means(kmeans=kmeans_rand, name="random", data=data, labels=labels)

    # PCA-based initialization
    pca = PCA(n_components=n_digits).fit(data)
    kmeans_pca = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
    res_pca = bench_k_means(kmeans=kmeans_pca, name="PCA-based", data=data, labels=labels)

    print(82 * "_")

    # Create PCA visualization (use k-means++ on reduced data)
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans_viz = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
    kmeans_viz.fit(reduced_data)

    return {
        "k-means++": res_kpp,
        "random": res_rand,
        "PCA-based": res_pca,
        "reduced_data": reduced_data,
        "kmeans_viz": kmeans_viz,
    }


# =========================================================================
# XORQ WAY -- deferred fit/predict, deferred metrics
# =========================================================================


def xorq_way(dataset):
    """Deferred xorq: wrap KMeans pipelines in Pipeline.from_instance,
    fit/predict deferred, compute deferred clustering metrics, create
    deferred PCA visualization.

    Returns dict with deferred metric expressions and plot expression.
    """
    import pandas as pd

    data = dataset["data"]
    labels = dataset["labels"]
    n_digits = dataset["n_digits"]

    con = xo.connect()

    # Register full dataset with labels
    # Use 'f' prefix for feature columns to avoid numeric-only names
    df = pd.DataFrame(data, columns=[f"f{i}" for i in range(data.shape[1])])
    df["label"] = labels
    table = con.register(df, "digits")

    # Feature columns (all except label)
    features = tuple(f"f{i}" for i in range(data.shape[1]))

    results = {}

    # k-means++ initialization
    sklearn_pipe_kpp = make_pipeline(
        StandardScaler(),
        KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
    )
    xorq_pipe_kpp = Pipeline.from_instance(sklearn_pipe_kpp)
    fitted_kpp = xorq_pipe_kpp.fit(table, features=features, target="label")
    preds_kpp = fitted_kpp.predict(table, name="pred")

    # Deferred metrics for k-means++
    make_metric = deferred_sklearn_metric(target="label", pred="pred")
    metrics_kpp = preds_kpp.agg(
        homogeneity=make_metric(metric=metrics.homogeneity_score),
        completeness=make_metric(metric=metrics.completeness_score),
        v_measure=make_metric(metric=metrics.v_measure_score),
        ari=make_metric(metric=metrics.adjusted_rand_score),
        ami=make_metric(metric=metrics.adjusted_mutual_info_score),
    )
    results["k-means++"] = {"metrics": metrics_kpp}

    # Random initialization
    sklearn_pipe_rand = make_pipeline(
        StandardScaler(),
        KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)
    )
    xorq_pipe_rand = Pipeline.from_instance(sklearn_pipe_rand)
    fitted_rand = xorq_pipe_rand.fit(table, features=features, target="label")
    preds_rand = fitted_rand.predict(table, name="pred")

    metrics_rand = preds_rand.agg(
        homogeneity=make_metric(metric=metrics.homogeneity_score),
        completeness=make_metric(metric=metrics.completeness_score),
        v_measure=make_metric(metric=metrics.v_measure_score),
        ari=make_metric(metric=metrics.adjusted_rand_score),
        ami=make_metric(metric=metrics.adjusted_mutual_info_score),
    )
    results["random"] = {"metrics": metrics_rand}

    # PCA-based initialization
    # Note: PCA-based init uses pca.components_ (numpy array) which is unhashable.
    # Xorq's frozen classes require hashable parameters, so we skip this for xorq.
    # This is a known limitation when wrapping sklearn estimators with array parameters.
    # For demonstration, we include this in sklearn_way only.
    print("Note: PCA-based initialization skipped for xorq due to unhashable numpy array parameter.")

    # Return table for deferred plotting in main()
    results["table"] = table

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    dataset = _load_data()

    print(f"# digits: {dataset['n_digits']}; # samples: {dataset['n_samples']}; # features {dataset['n_features']}")
    print()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(dataset)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(dataset)

    # Execute deferred metrics and assert equivalence
    print("\n=== ASSERTIONS ===")
    print("Comparing clustering metrics (sklearn vs xorq):")
    print("(Note: PCA-based initialization only tested with sklearn due to unhashable parameter constraint)")

    # Only compare k-means++ and random for xorq
    for init_method in ["k-means++", "random"]:
        sk_res = sk_results[init_method]
        xo_metrics_df = xo_results[init_method]["metrics"].execute()

        print(f"\n{init_method}:")
        print(f"  homogeneity  - sklearn: {sk_res['homogeneity']:.3f}, xorq: {xo_metrics_df['homogeneity'].iloc[0]:.3f}")
        print(f"  completeness - sklearn: {sk_res['completeness']:.3f}, xorq: {xo_metrics_df['completeness'].iloc[0]:.3f}")
        print(f"  v_measure    - sklearn: {sk_res['v_measure']:.3f}, xorq: {xo_metrics_df['v_measure'].iloc[0]:.3f}")
        print(f"  ARI          - sklearn: {sk_res['ari']:.3f}, xorq: {xo_metrics_df['ari'].iloc[0]:.3f}")
        print(f"  AMI          - sklearn: {sk_res['ami']:.3f}, xorq: {xo_metrics_df['ami'].iloc[0]:.3f}")

        # Assert equivalence
        np.testing.assert_allclose(sk_res['homogeneity'], xo_metrics_df['homogeneity'].iloc[0], rtol=1e-2)
        np.testing.assert_allclose(sk_res['completeness'], xo_metrics_df['completeness'].iloc[0], rtol=1e-2)
        np.testing.assert_allclose(sk_res['v_measure'], xo_metrics_df['v_measure'].iloc[0], rtol=1e-2)
        np.testing.assert_allclose(sk_res['ari'], xo_metrics_df['ari'].iloc[0], rtol=1e-2)
        np.testing.assert_allclose(sk_res['ami'], xo_metrics_df['ami'].iloc[0], rtol=1e-2)

    # Also print PCA-based sklearn results for reference
    sk_res_pca = sk_results["PCA-based"]
    print(f"\nPCA-based (sklearn only):")
    print(f"  homogeneity  - sklearn: {sk_res_pca['homogeneity']:.3f}")
    print(f"  completeness - sklearn: {sk_res_pca['completeness']:.3f}")
    print(f"  v_measure    - sklearn: {sk_res_pca['v_measure']:.3f}")
    print(f"  ARI          - sklearn: {sk_res_pca['ari']:.3f}")
    print(f"  AMI          - sklearn: {sk_res_pca['ami']:.3f}")

    print("\nAssertions passed: sklearn and xorq clustering metrics match for k-means++ and random initialization.")

    # Execute deferred plot in main()
    print("\n=== PLOTTING ===")
    xo_png = deferred_matplotlib_plot(xo_results["table"], _build_pca_plot).execute()

    # Build sklearn plot
    sk_fig = _plot_pca_clusters(
        sk_results["reduced_data"],
        sk_results["kmeans_viz"],
        dataset["n_digits"]
    )

    # Composite: sklearn (left) | xorq (right)
    xo_img = load_plot_bytes(xo_png)
    sk_img = fig_to_image(sk_fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(sk_img)
    axes[0].set_title("sklearn")
    axes[0].axis("off")

    axes[1].imshow(xo_img)
    axes[1].set_title("xorq")
    axes[1].axis("off")

    plt.suptitle("K-Means Clustering on Digits: sklearn vs xorq", fontsize=14)
    plt.tight_layout()
    out = "imgs/plot_kmeans_digits.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Composite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
