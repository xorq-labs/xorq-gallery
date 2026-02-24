"""K-Means clustering on handwritten digits
===========================================

sklearn: Load digits dataset, compare three KMeans initialization strategies
(k-means++, random, PCA-based) using StandardScaler preprocessing, evaluate
with homogeneity, completeness, V-measure, ARI, AMI, and silhouette metrics,
visualize PCA-reduced clusters.

xorq: Same KMeans pipelines wrapped in Pipeline.from_instance, deferred
fit/predict, deferred metrics evaluation, deferred PCA reduction and plotting.
PCA-based initialization is skipped because KMeans(init=pca.components_) passes
a numpy array, which is unhashable for xorq's frozen parameter storage.

Both produce identical clustering metrics (for k-means++ and random init).

Dataset: load_digits (sklearn handwritten digits 0-9)
"""

from __future__ import annotations

import os
from time import time

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

PRED_COL = "pred"
LABEL_COL = "label"
FEATURE_PREFIX = "f"

# Clustering metrics to evaluate
CLUSTERING_METRICS = (
    metrics.homogeneity_score,
    metrics.completeness_score,
    metrics.v_measure_score,
    metrics.adjusted_rand_score,
    metrics.adjusted_mutual_info_score,
)

# Metric names for results dict
METRIC_NAMES = ("homogeneity", "completeness", "v_measure", "ari", "ami")


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
    x_min = reduced_data[:, 0].min() - 1
    x_max = reduced_data[:, 0].max() + 1
    y_min = reduced_data[:, 1].min() - 1
    y_max = reduced_data[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

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

    ax.plot(
        reduced_data[:, 0],
        reduced_data[:, 1],
        "k.",
        markersize=2
    )

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

    fig.tight_layout()
    return fig


def _build_pca_plot(df):
    """Build PCA cluster visualization from deferred execution.

    This function receives the full dataset, refits KMeans on PCA-reduced
    data (similar to sklearn approach), and creates the visualization.
    """
    # Extract data from dataframe
    data_cols = [col for col in df.columns if col.startswith(FEATURE_PREFIX)]
    data = df[data_cols].values
    n_digits = df[LABEL_COL].nunique()

    # Reduce to 2D via PCA
    reduced_data = PCA(n_components=2).fit_transform(data)

    # Refit KMeans on reduced data for visualization
    kmeans_viz = KMeans(
        init="k-means++",
        n_clusters=n_digits,
        n_init=4,
        random_state=0
    )
    kmeans_viz.fit(reduced_data)

    return _plot_pca_clusters(reduced_data, kmeans_viz, n_digits)


# ---------------------------------------------------------------------------
# Benchmarking helper
# ---------------------------------------------------------------------------


def _bench_k_means(kmeans, name, data, labels):
    """Benchmark KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        KMeans estimator to benchmark
    name : str
        Name of the initialization method
    data : ndarray
        Input data
    labels : ndarray
        True labels

    Returns
    -------
    dict
        Benchmark results including metrics and fitted estimator
    """
    t0 = time()
    estimator = SklearnPipeline([("standardscaler", StandardScaler()), ("kmeans", kmeans)]).fit(data)
    fit_time = time() - t0

    # Basic results
    results = [name, fit_time, estimator[-1].inertia_]

    # Clustering metrics
    results += [
        metric_fn(labels, estimator[-1].labels_)
        for metric_fn in CLUSTERING_METRICS
    ]

    # Silhouette score
    silhouette = metrics.silhouette_score(
        data,
        estimator[-1].labels_,
        metric="euclidean",
        sample_size=300,
    )
    results += [silhouette]

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


# =========================================================================
# SKLEARN WAY -- eager fit/predict, benchmarking
# =========================================================================


def sklearn_way(dataset):
    """Eager sklearn: benchmark three KMeans initialization strategies,
    compute clustering metrics, create PCA visualization."""
    data = dataset["data"]
    labels = dataset["labels"]
    n_digits = dataset["n_digits"]

    print(82 * "_")
    print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

    # k-means++ initialization
    kmeans_kpp = KMeans(
        init="k-means++",
        n_clusters=n_digits,
        n_init=4,
        random_state=0
    )
    res_kpp = _bench_k_means(
        kmeans=kmeans_kpp,
        name="k-means++",
        data=data,
        labels=labels
    )

    # Random initialization
    kmeans_rand = KMeans(
        init="random",
        n_clusters=n_digits,
        n_init=4,
        random_state=0
    )
    res_rand = _bench_k_means(
        kmeans=kmeans_rand,
        name="random",
        data=data,
        labels=labels
    )

    # PCA-based initialization
    pca = PCA(n_components=n_digits).fit(data)
    kmeans_pca = KMeans(
        init=pca.components_,
        n_clusters=n_digits,
        n_init=1
    )
    res_pca = _bench_k_means(
        kmeans=kmeans_pca,
        name="PCA-based",
        data=data,
        labels=labels
    )

    print(82 * "_")

    # Create PCA visualization (use k-means++ on reduced data)
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans_viz = KMeans(
        init="k-means++",
        n_clusters=n_digits,
        n_init=4,
        random_state=0
    )
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
    fit/predict deferred, compute deferred clustering metrics.

    Returns dict with deferred metric expressions and table for plotting.
    """
    data = dataset["data"]
    labels = dataset["labels"]
    n_digits = dataset["n_digits"]

    con = xo.connect()

    # Register full dataset with labels
    # Use 'f' prefix for feature columns to avoid numeric-only names
    df = pd.DataFrame(
        data,
        columns=[f"{FEATURE_PREFIX}{i}" for i in range(data.shape[1])]
    )
    df[LABEL_COL] = labels
    table = con.register(df, "digits")

    # Feature columns (all except label)
    features = tuple(f"{FEATURE_PREFIX}{i}" for i in range(data.shape[1]))

    make_metric = deferred_sklearn_metric(target=LABEL_COL, pred=PRED_COL)

    # k-means++ initialization
    sklearn_pipe_kpp = SklearnPipeline([
        ("standardscaler", StandardScaler()),
        ("kmeans", KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)),
    ])
    xorq_pipe_kpp = Pipeline.from_instance(sklearn_pipe_kpp)
    fitted_kpp = xorq_pipe_kpp.fit(table, features=features, target=LABEL_COL)
    preds_kpp = fitted_kpp.predict(table, name=PRED_COL)
    metrics_kpp = preds_kpp.agg(**{
        metric_name: make_metric(metric=metric_fn)
        for metric_name, metric_fn in zip(METRIC_NAMES, CLUSTERING_METRICS)
    })

    # Random initialization
    sklearn_pipe_rand = SklearnPipeline([
        ("standardscaler", StandardScaler()),
        ("kmeans", KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)),
    ])
    xorq_pipe_rand = Pipeline.from_instance(sklearn_pipe_rand)
    fitted_rand = xorq_pipe_rand.fit(table, features=features, target=LABEL_COL)
    preds_rand = fitted_rand.predict(table, name=PRED_COL)
    metrics_rand = preds_rand.agg(**{
        metric_name: make_metric(metric=metric_fn)
        for metric_name, metric_fn in zip(METRIC_NAMES, CLUSTERING_METRICS)
    })

    # Note: PCA-based init passes pca.components_ (a numpy ndarray) as the
    # `init` parameter to KMeans.  xorq's Pipeline stores model parameters in
    # attrs-based frozen classes that require all values to be hashable.
    # numpy arrays are not hashable, so Pipeline.from_instance raises a
    # TypeError when it tries to freeze the KMeans params.  Until xorq adds
    # automatic array-to-tuple coercion for init parameters, PCA-based init
    # must be handled outside the deferred pipeline.
    print(
        "Note: PCA-based initialization skipped for xorq -- "
        "KMeans(init=pca.components_) passes a numpy array, which is "
        "unhashable for xorq's frozen parameter storage."
    )

    return {
        "k-means++": metrics_kpp,
        "random": metrics_rand,
        "table": table,
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    dataset = _load_data()

    print(
        f"# digits: {dataset['n_digits']}; "
        f"# samples: {dataset['n_samples']}; "
        f"# features {dataset['n_features']}"
    )
    print()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(dataset)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(dataset)

    # Execute deferred metrics and build comparison dataframe
    print("\n=== ASSERTIONS ===")
    print("Comparing clustering metrics (sklearn vs xorq):")
    print(
        "(Note: PCA-based initialization only tested with sklearn "
        "due to unhashable parameter constraint)"
    )

    # Build metrics comparison DataFrame
    sklearn_metrics_data = []
    xorq_metrics_data = []

    init_methods = ("k-means++", "random")

    for init_method in init_methods:
        sk_res = sk_results[init_method]
        xo_metrics_df = xo_results[init_method].execute()

        print(f"\n{init_method}:")
        for metric_name in METRIC_NAMES:
            sk_value = sk_res[metric_name]
            xo_value = xo_metrics_df[metric_name].iloc[0]
            print(
                f"  {metric_name:12s} - "
                f"sklearn: {sk_value:.3f}, "
                f"xorq: {xo_value:.3f}"
            )

        sklearn_metrics_data.append({
            "init": init_method,
            **{name: sk_res[name] for name in METRIC_NAMES}
        })

        xorq_metrics_data.append({
            "init": init_method,
            **{
                name: xo_metrics_df[name].iloc[0]
                for name in METRIC_NAMES
            }
        })

    # Create DataFrames for comparison
    sklearn_metrics_df = pd.DataFrame(sklearn_metrics_data).set_index("init")
    xorq_metrics_df = pd.DataFrame(xorq_metrics_data).set_index("init")

    # Single assertion comparing all metrics
    pd.testing.assert_frame_equal(
        sklearn_metrics_df,
        xorq_metrics_df,
        rtol=1e-2,
        check_dtype=False
    )

    # Also print PCA-based sklearn results for reference
    sk_res_pca = sk_results["PCA-based"]
    print(f"\nPCA-based (sklearn only):")
    for metric_name in METRIC_NAMES:
        print(f"  {metric_name:12s} - sklearn: {sk_res_pca[metric_name]:.3f}")

    print(
        "\nAssertions passed: sklearn and xorq clustering metrics match "
        "for k-means++ and random initialization."
    )

    print("\n=== PLOTTING ===")
    xo_png = deferred_matplotlib_plot(
        xo_results["table"],
        _build_pca_plot
    ).execute()

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

    fig.suptitle(
        "K-Means Clustering on Digits: sklearn vs xorq",
        fontsize=14
    )
    fig.tight_layout()
    out = "imgs/plot_kmeans_digits.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Composite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
