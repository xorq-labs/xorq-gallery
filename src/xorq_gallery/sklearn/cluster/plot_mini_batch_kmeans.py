"""Comparison of the K-Means and MiniBatchKMeans clustering algorithms
=====================================================================

sklearn: Generate synthetic 2D blob data (3000 samples, 3 centers), fit both
KMeans and MiniBatchKMeans, visualise cluster assignments and centroids,
and highlight points that are labelled differently between the two algorithms.

xorq: Same clustering pipelines wrapped in Pipeline.from_instance, deferred
fit/predict.  Predictions are executed in main() and plotted from the
materialised results.
apply_deterministic_sort imposes a stable row order on the ibis table; the
pandas side is derived from the sorted table so both paths see identical data
(MiniBatchKMeans mini-batch selection is order-sensitive).

MiniBatchKMeans is an alternative online implementation that does incremental
updates of the centres positions using mini-batches. For large-scale learning
it is much faster than the default batch implementation, at the cost of
slightly different results.

Dataset: make_blobs (sklearn synthetic, 3000 samples, 3 centers)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.pipeline import Pipeline as SklearnPipeline
from xorq.expr.ml.cross_validation import apply_deterministic_sort
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import fig_to_image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 0
N_SAMPLES = 3000
N_CLUSTERS = 3
CENTERS = ((1, 1), (-1, -1), (1, -1))
CLUSTER_STD = 0.7
BATCH_SIZE = 45
N_INIT = 10
MAX_NO_IMPROVEMENT = 10
FEATURES = ("f0", "f1")
TARGET_COL = "label"
PRED_COL = "pred"

PLOT_COLORS = ("#4EACC5", "#FF9C34", "#4E9A06")


# ---------------------------------------------------------------------------
# Estimator builders (hashable-friendly: recreate from constants)
# ---------------------------------------------------------------------------


def _build_kmeans():
    """Build a KMeans estimator from module-level constants."""
    return KMeans(
        init="k-means++",
        n_clusters=N_CLUSTERS,
        n_init=N_INIT,
        random_state=RANDOM_STATE,
    )


def _build_mbk():
    """Build a MiniBatchKMeans estimator from module-level constants."""
    return MiniBatchKMeans(
        init="k-means++",
        n_clusters=N_CLUSTERS,
        n_init=N_INIT,
        batch_size=BATCH_SIZE,
        max_no_improvement=MAX_NO_IMPROVEMENT,
        random_state=RANDOM_STATE,
    )


ALGORITHM_SPECS = (
    ("KMeans", "kmeans", _build_kmeans),
    ("MiniBatchKMeans", "minibatchkmeans", _build_mbk),
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data():
    """Generate synthetic 2D blobs with deterministic row order.

    Returns both a pandas DataFrame and an ibis table in the same
    deterministic row order.
    apply_deterministic_sort imposes a stable row order on the ibis table;
    the pandas side is derived from the sorted table so both paths see
    identical data (MiniBatchKMeans mini-batch selection is order-sensitive).
    """
    X, y = make_blobs(
        n_samples=N_SAMPLES,
        centers=list(CENTERS),
        cluster_std=CLUSTER_STD,
        random_state=RANDOM_STATE,
    )
    df = pd.DataFrame(X, columns=list(FEATURES))
    df[TARGET_COL] = y

    con = xo.connect()
    raw_table = con.register(df, "blobs_raw")
    table = apply_deterministic_sort(raw_table)

    # Materialise sorted order back to pandas so sklearn sees the same rows
    df = table.execute()

    return df, table


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_results(X, km_labels, km_centers, mbk_labels, mbk_centers, title_prefix=""):
    """Build the 3-panel comparison figure.

    Panel 1: KMeans clustering with cluster centres
    Panel 2: MiniBatchKMeans clustering with cluster centres
    Panel 3: Points labelled differently between the two algorithms

    Parameters
    ----------
    X : ndarray, shape (n_samples, 2)
        Feature array.
    km_labels, mbk_labels : ndarray
        Cluster assignments from KMeans and MiniBatchKMeans.
    km_centers, mbk_centers : ndarray
        Cluster centres from KMeans and MiniBatchKMeans.
    title_prefix : str
        Prefix for the suptitle (e.g. "sklearn: " or "xorq: ").

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: KMeans ---
    ax = axes[0]
    for k, col in enumerate(PLOT_COLORS):
        mask = km_labels == k
        ax.scatter(X[mask, 0], X[mask, 1], c=col, marker=".", s=6, alpha=0.6)
    ax.scatter(
        km_centers[:, 0],
        km_centers[:, 1],
        marker="o",
        c="w",
        edgecolor="k",
        s=100,
        linewidths=2,
        zorder=10,
    )
    ax.set_title("KMeans")
    ax.set_xticks(())
    ax.set_yticks(())

    # --- Panel 2: MiniBatchKMeans ---
    ax = axes[1]
    for k, col in enumerate(PLOT_COLORS):
        mask = mbk_labels == k
        ax.scatter(X[mask, 0], X[mask, 1], c=col, marker=".", s=6, alpha=0.6)
    ax.scatter(
        mbk_centers[:, 0],
        mbk_centers[:, 1],
        marker="o",
        c="w",
        edgecolor="k",
        s=100,
        linewidths=2,
        zorder=10,
    )
    ax.set_title("MiniBatchKMeans")
    ax.set_xticks(())
    ax.set_yticks(())

    # --- Panel 3: Difference ---
    ax = axes[2]
    # Find which MiniBatchKMeans centre is closest to each KMeans centre
    order = pairwise_distances_argmin(km_centers, mbk_centers)
    mbk_labels_reordered = np.empty_like(mbk_labels)
    for new_k, old_k in enumerate(order):
        mbk_labels_reordered[mbk_labels == old_k] = new_k

    different = km_labels != mbk_labels_reordered
    ax.scatter(
        X[~different, 0],
        X[~different, 1],
        c="#bbbbbb",
        marker=".",
        s=6,
        alpha=0.3,
        label="Same label",
    )
    ax.scatter(
        X[different, 0],
        X[different, 1],
        c="m",
        marker=".",
        s=6,
        alpha=0.8,
        label="Different label",
    )
    n_diff = different.sum()
    ax.set_title(f"Difference: {n_diff} ({100 * n_diff / len(X):.1f}%)")
    ax.legend(loc="best", fontsize=8, markerscale=3)
    ax.set_xticks(())
    ax.set_yticks(())

    fig.suptitle(
        f"{title_prefix}K-Means vs MiniBatchKMeans",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def _remap_labels(reference, labels):
    """Remap *labels* so that cluster IDs match *reference* as closely as
    possible (maximise overlap via the Hungarian algorithm).
    """
    ref_ids = np.unique(reference)
    lab_ids = np.unique(labels)

    cost = np.zeros((len(ref_ids), len(lab_ids)), dtype=int)
    for i, r in enumerate(ref_ids):
        for j, lab in enumerate(lab_ids):
            cost[i, j] = -np.sum((reference == r) & (labels == lab))

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {lab_ids[c]: ref_ids[r] for r, c in zip(row_ind, col_ind)}

    next_id = ref_ids.max() + 1 if len(ref_ids) else 0
    out = np.empty_like(labels)
    for idx, val in enumerate(labels):
        if val in mapping:
            out[idx] = mapping[val]
        else:
            out[idx] = next_id
            next_id += 1

    return out


def _centroids_from_labels(X, labels, n_clusters):
    """Compute cluster centroids from data and label assignments."""
    return np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])


# =========================================================================
# SKLEARN WAY -- eager fit
# =========================================================================


def sklearn_way(df):
    """Eager sklearn: fit KMeans and MiniBatchKMeans on the blob data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns f0, f1.

    Returns
    -------
    dict with algo_name -> {"labels", "centers", "inertia"}.
    """
    X = df[list(FEATURES)].values
    results = {}

    for algo_name, step_name, builder in ALGORITHM_SPECS:
        pipeline = SklearnPipeline([(step_name, builder())])
        pipeline.fit(X)
        estimator = pipeline.named_steps[step_name]

        results[algo_name] = {
            "labels": estimator.labels_,
            "centers": estimator.cluster_centers_,
            "inertia": estimator.inertia_,
        }
        print(f"  {algo_name:20s}  inertia={estimator.inertia_:.1f}")

    return results


# =========================================================================
# XORQ WAY -- deferred fit/predict
# =========================================================================


def xorq_way(table):
    """Deferred xorq: Pipeline.from_instance + fit + predict for each algo.

    Returns dict of algo_name -> deferred prediction expression.
    No .execute() calls here.
    """
    preds = {}

    for algo_name, step_name, builder in ALGORITHM_SPECS:
        sklearn_pipe = SklearnPipeline([(step_name, builder())])
        xorq_pipe = Pipeline.from_instance(sklearn_pipe)
        fitted = xorq_pipe.fit(table, features=FEATURES, target=TARGET_COL)
        pred_expr = fitted.predict(table, name=PRED_COL)

        preds[algo_name] = pred_expr

    return preds


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    print("=== GENERATING DATA ===")
    df, table = _load_data()
    print(f"Generated {len(df)} samples")

    print("\n=== SKLEARN WAY ===")
    sk_results = sklearn_way(df)

    print("\n=== XORQ WAY ===")
    xo_preds = xorq_way(table)

    # --- Execute deferred predictions and assert equivalence ---
    print("\n=== ASSERTIONS ===")
    X = df[list(FEATURES)].values

    xo_results = {}
    for algo_name, _, _ in ALGORITHM_SPECS:
        sk_labels = sk_results[algo_name]["labels"]
        xo_labels = xo_preds[algo_name].execute()[PRED_COL].values

        sk_n = len(np.unique(sk_labels))
        xo_n = len(np.unique(xo_labels))
        assert sk_n == xo_n, (
            f"{algo_name}: cluster count mismatch (sklearn={sk_n}, xorq={xo_n})"
        )

        # Remap xorq labels to match sklearn colours for plotting
        xo_remapped = _remap_labels(sk_labels, xo_labels)
        xo_results[algo_name] = {
            "labels": xo_remapped,
            "centers": _centroids_from_labels(X, xo_remapped, N_CLUSTERS),
        }

        overlap = np.mean(sk_labels == xo_remapped)
        print(
            f"  {algo_name:20s}  clusters: sklearn={sk_n}, xorq={xo_n}  "
            f"overlap={overlap:.2%}  OK"
        )

    print("Assertions passed: sklearn and xorq cluster counts match.")

    # --- Plotting ---
    print("\n=== PLOTTING ===")

    sk_fig = _plot_results(
        X,
        sk_results["KMeans"]["labels"],
        sk_results["KMeans"]["centers"],
        sk_results["MiniBatchKMeans"]["labels"],
        sk_results["MiniBatchKMeans"]["centers"],
        title_prefix="sklearn: ",
    )
    xo_fig = _plot_results(
        X,
        xo_results["KMeans"]["labels"],
        xo_results["KMeans"]["centers"],
        xo_results["MiniBatchKMeans"]["labels"],
        xo_results["MiniBatchKMeans"]["centers"],
        title_prefix="xorq: ",
    )

    # Composite: sklearn (top) | xorq (bottom)
    fig, axes = plt.subplots(2, 1, figsize=(16, 11))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(
        "K-Means vs MiniBatchKMeans: sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    out = "imgs/plot_mini_batch_kmeans.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
