"""A demo of the mean-shift clustering algorithm
=================================================

sklearn: Generate synthetic 2D blob data (10 000 samples, 3 centers),
estimate the kernel bandwidth with estimate_bandwidth, fit MeanShift with
bin_seeding, and visualize the discovered clusters and centroids.

xorq: Same MeanShift pipeline wrapped in Pipeline.from_instance, deferred
fit/predict.  The bandwidth is pre-computed eagerly (it is a data-dependent
hyperparameter, not a learned model parameter) and passed as a fixed
constructor argument to MeanShift.
apply_deterministic_sort imposes a stable row order on the ibis table; the
pandas side is derived from the sorted table so both paths see identical data
(MeanShift with bin_seeding is order-sensitive).

MeanShift is a centroid-based algorithm that discovers blobs in a smooth
density of samples without requiring the number of clusters to be specified
in advance.

Dataset: make_blobs (sklearn synthetic, 10 000 samples, 3 centers)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline as SklearnPipeline
from xorq.expr.ml.cross_validation import apply_deterministic_sort
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import fig_to_image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 0
N_SAMPLES = 10_000
CENTERS = ((1, 1), (-1, -1), (1, -1))
CLUSTER_STD = 0.6
BANDWIDTH_QUANTILE = 0.2
BANDWIDTH_N_SAMPLES = 500
FEATURES = ("f0", "f1")
TARGET_COL = "label"
PRED_COL = "pred"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data():
    """Generate synthetic 2D blobs with deterministic row order.

    Returns both a pandas DataFrame and an ibis table in the same
    deterministic row order, plus the estimated bandwidth.
    apply_deterministic_sort imposes a stable row order on the ibis table;
    the pandas side is derived from the sorted table so both paths see
    identical data (MeanShift with bin_seeding is order-sensitive).
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

    bandwidth = estimate_bandwidth(
        df[list(FEATURES)].values,
        quantile=BANDWIDTH_QUANTILE,
        n_samples=BANDWIDTH_N_SAMPLES,
    )
    return df, table, bandwidth


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_clusters(X, labels, cluster_centers, title_prefix=""):
    """Scatter plot of cluster assignments with centroids.

    Parameters
    ----------
    X : ndarray, shape (n_samples, 2)
        Feature array.
    labels : ndarray
        Cluster assignments per sample.
    cluster_centers : ndarray
        Cluster centroid coordinates.
    title_prefix : str
        Prefix for the plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

    for k, color in zip(labels_unique, colors):
        mask = labels == k
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=6,
            c=[color],
            alpha=0.6,
            label=f"Cluster {k}",
        )

    ax.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        marker="x",
        s=120,
        linewidths=3,
        color="r",
        zorder=10,
        label="Centroids",
    )

    ax.set_title(
        f"{title_prefix}Estimated number of clusters: {n_clusters}",
        fontsize=12,
    )
    ax.legend(loc="best", fontsize=8, markerscale=2)
    ax.set_xticks(())
    ax.set_yticks(())
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


def _centroids_from_labels(X, labels):
    """Compute cluster centroids from data and label assignments."""
    return np.array([X[labels == k].mean(axis=0) for k in np.unique(labels)])


# =========================================================================
# SKLEARN WAY -- eager fit
# =========================================================================


def sklearn_way(df, bandwidth):
    """Eager sklearn: fit MeanShift on the blob data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns f0, f1.
    bandwidth : float
        Pre-estimated kernel bandwidth.

    Returns
    -------
    dict with "labels" and "centers".
    """
    X = df[list(FEATURES)].values

    pipeline = SklearnPipeline(
        [("meanshift", MeanShift(bandwidth=bandwidth, bin_seeding=True))]
    )
    pipeline.fit(X)
    estimator = pipeline.named_steps["meanshift"]

    return {
        "labels": estimator.labels_,
        "centers": estimator.cluster_centers_,
    }


# =========================================================================
# XORQ WAY -- deferred fit/predict
# =========================================================================


def xorq_way(table, bandwidth):
    """Deferred xorq: Pipeline.from_instance + fit + predict.

    Returns a deferred prediction expression. No .execute() calls.
    """
    sklearn_pipe = SklearnPipeline(
        [("meanshift", MeanShift(bandwidth=bandwidth, bin_seeding=True))]
    )
    xorq_pipe = Pipeline.from_instance(sklearn_pipe)
    fitted = xorq_pipe.fit(table, features=FEATURES, target=TARGET_COL)
    pred_expr = fitted.predict(table, name=PRED_COL)

    return pred_expr


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    print("=== GENERATING DATA ===")
    df, table, bandwidth = _load_data()
    print(f"Generated {len(df)} samples, estimated bandwidth = {bandwidth:.4f}")

    print("\n=== SKLEARN WAY ===")
    sk_result = sklearn_way(df, bandwidth)
    sk_labels = sk_result["labels"]
    sk_centers = sk_result["centers"]
    n_clusters_sk = len(np.unique(sk_labels))
    print(f"  Number of estimated clusters: {n_clusters_sk}")

    print("\n=== XORQ WAY ===")
    xo_pred_expr = xorq_way(table, bandwidth)

    # --- Execute deferred predictions and assert equivalence ---
    print("\n=== ASSERTIONS ===")
    X = df[list(FEATURES)].values
    xo_labels = xo_pred_expr.execute()[PRED_COL].values

    n_clusters_xo = len(np.unique(xo_labels))
    assert n_clusters_sk == n_clusters_xo, (
        f"Cluster count mismatch: sklearn={n_clusters_sk}, xorq={n_clusters_xo}"
    )

    # Remap xorq labels to match sklearn colours for plotting
    xo_remapped = _remap_labels(sk_labels, xo_labels)
    xo_centers = _centroids_from_labels(X, xo_remapped)

    print(f"  sklearn clusters: {n_clusters_sk}, xorq clusters: {n_clusters_xo}  OK")
    print("Assertions passed: sklearn and xorq cluster counts match.")

    # --- Plotting ---
    print("\n=== PLOTTING ===")

    sk_fig = _plot_clusters(X, sk_labels, sk_centers, title_prefix="sklearn: ")
    xo_fig = _plot_clusters(X, xo_remapped, xo_centers, title_prefix="xorq: ")

    # Composite: sklearn (left) | xorq (right)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(
        "Mean-Shift Clustering: sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    out = "imgs/plot_mean_shift.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
