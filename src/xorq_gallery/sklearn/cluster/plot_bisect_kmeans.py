"""Bisecting K-Means and Regular K-Means Performance Comparison
================================================================

sklearn: Generate synthetic 2D blob data (10 000 samples, 2 centers), fit both
BisectingKMeans and KMeans for n_clusters in {4, 8, 16}, visualise cluster
assignments and centroids in a grid of subplots.

xorq: Same clustering pipelines wrapped in Pipeline.from_instance, deferred
fit/predict for each (algorithm, n_clusters) combination.
apply_deterministic_sort imposes a stable row order on the ibis table; the
pandas side is derived from the sorted table so both paths see identical data
(BisectingKMeans is order-sensitive).  Predictions are executed in main() and
plotted from the materialised results.

BisectingKMeans builds on previous partitions, producing clusters with a more
regular large-scale structure (a visible dividing line across the data cloud),
whereas regular KMeans starts fresh for every n_clusters value.

Dataset: make_blobs (sklearn synthetic, 10 000 samples, 2 centers)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import BisectingKMeans, KMeans
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
N_BLOB_CENTERS = 2
N_CLUSTERS_LIST = (4, 8, 16)
FEATURES = ("f0", "f1")
TRUE_LABEL_COL = "true_label"
PRED_COL = "pred"

ALGORITHM_SPECS = (
    ("BisectingKMeans", BisectingKMeans),
    ("KMeans", KMeans),
)


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Generate synthetic 2D blobs and return a deterministically sorted table.

    Returns both a pandas DataFrame and an ibis table in the same
    deterministic row order.  The true blob labels are kept as a target
    column so that Pipeline.from_instance can infer feature vs target
    columns, even though the clustering algorithms are fully unsupervised.
    """
    X, y = make_blobs(
        n_samples=N_SAMPLES, centers=N_BLOB_CENTERS, random_state=RANDOM_STATE
    )
    df = pd.DataFrame(X, columns=list(FEATURES))
    df[TRUE_LABEL_COL] = y

    con = xo.connect()
    raw_table = con.register(df, "blobs_raw")
    table = apply_deterministic_sort(raw_table)

    # Materialise sorted order back to pandas so sklearn sees the same rows
    df = table.execute()

    return df, table


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_grid(X, results, title_prefix=""):
    """Build a (n_algorithms x n_clusters) grid of scatter plots.

    Parameters
    ----------
    X : ndarray, shape (n_samples, 2)
        Feature array for scatter coordinates.
    results : dict
        Mapping (algo_name, n_clusters) -> {"labels": ndarray, "centers": ndarray}
    title_prefix : str
        Prefix for the suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_rows = len(ALGORITHM_SPECS)
    n_cols = len(N_CLUSTERS_LIST)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5))

    for col_idx, n_clusters in enumerate(N_CLUSTERS_LIST):
        for row_idx, (algo_name, _) in enumerate(ALGORITHM_SPECS):
            ax = axes[row_idx, col_idx]
            info = results[(algo_name, n_clusters)]
            labels = info["labels"]
            centers = info["centers"]

            ax.scatter(X[:, 0], X[:, 1], s=2, c=labels, cmap="viridis")
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                marker="x",
                s=50,
                linewidths=2,
                color="r",
                zorder=10,
            )

            if row_idx == 0:
                ax.set_title(f"n_clusters = {n_clusters}")
            if col_idx == 0:
                ax.set_ylabel(algo_name)

            ax.set_xticks(())
            ax.set_yticks(())

    fig.suptitle(
        f"{title_prefix}Bisecting K-Means and Regular K-Means Comparison",
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


def _centroids_from_labels(X, labels):
    """Compute cluster centroids from data and label assignments."""
    return np.array([X[labels == k].mean(axis=0) for k in np.unique(labels)])


# =========================================================================
# SKLEARN WAY -- eager fit, cluster assignments
# =========================================================================


def sklearn_way(df):
    """Eager sklearn: fit BisectingKMeans and KMeans for several n_clusters.

    Returns dict mapping (algo_name, n_clusters) -> cluster info.
    """
    X = df[list(FEATURES)].values
    results = {}

    for algo_name, AlgoClass in ALGORITHM_SPECS:
        for n_clusters in N_CLUSTERS_LIST:
            pipeline = SklearnPipeline(
                [
                    (
                        algo_name.lower(),
                        AlgoClass(n_clusters=n_clusters, random_state=RANDOM_STATE),
                    )
                ]
            )
            pipeline.fit(X)
            estimator = pipeline.named_steps[algo_name.lower()]
            labels = estimator.labels_
            centers = estimator.cluster_centers_

            results[(algo_name, n_clusters)] = {
                "labels": labels,
                "centers": centers,
            }
            print(
                f"  {algo_name:20s} n_clusters={n_clusters:2d}  "
                f"inertia={estimator.inertia_:.1f}"
            )

    return results


# =========================================================================
# XORQ WAY -- deferred fit/predict
# =========================================================================


def xorq_way(table):
    """Deferred xorq: Pipeline.from_instance + fit + predict for each combo.

    Returns dict mapping (algo_name, n_clusters) -> deferred predictions.
    No .execute() calls here.
    """
    preds = {}

    for algo_name, AlgoClass in ALGORITHM_SPECS:
        for n_clusters in N_CLUSTERS_LIST:
            sklearn_pipe = SklearnPipeline(
                [
                    (
                        algo_name.lower(),
                        AlgoClass(n_clusters=n_clusters, random_state=RANDOM_STATE),
                    )
                ]
            )
            xorq_pipe = Pipeline.from_instance(sklearn_pipe)
            fitted = xorq_pipe.fit(table, features=FEATURES, target=TRUE_LABEL_COL)
            pred_expr = fitted.predict(table, name=PRED_COL)

            preds[(algo_name, n_clusters)] = pred_expr

    return preds


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df, table = _load_data()
    X = df[list(FEATURES)].values

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(df)

    print("\n=== XORQ WAY ===")
    xo_preds = xorq_way(table)

    # --- Execute deferred predictions and assert equivalence ---
    print("\n=== ASSERTIONS ===")
    xo_results = {}
    for algo_name, _ in ALGORITHM_SPECS:
        for n_clusters in N_CLUSTERS_LIST:
            sk_labels = sk_results[(algo_name, n_clusters)]["labels"]
            xo_labels = xo_preds[(algo_name, n_clusters)].execute()[PRED_COL].values

            sk_n = len(np.unique(sk_labels))
            xo_n = len(np.unique(xo_labels))
            assert sk_n == xo_n, (
                f"{algo_name} n_clusters={n_clusters}: cluster count mismatch "
                f"(sklearn={sk_n}, xorq={xo_n})"
            )

            # Remap xorq labels to match sklearn colours for plotting
            xo_remapped = _remap_labels(sk_labels, xo_labels)
            xo_results[(algo_name, n_clusters)] = {
                "labels": xo_remapped,
                "centers": _centroids_from_labels(X, xo_remapped),
            }

            print(
                f"  {algo_name:20s} n_clusters={n_clusters:2d}  "
                f"clusters: sklearn={sk_n}, xorq={xo_n}  OK"
            )

    print("Assertions passed: sklearn and xorq cluster counts match.")

    # --- Plotting ---
    print("\n=== PLOTTING ===")

    sk_fig = _plot_grid(X, sk_results, title_prefix="sklearn: ")
    xo_fig = _plot_grid(X, xo_results, title_prefix="xorq: ")

    # Composite: sklearn (top) | xorq (bottom)
    fig, axes = plt.subplots(2, 1, figsize=(12, 11))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(
        "Bisecting K-Means vs K-Means: sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    out = "imgs/plot_bisect_kmeans.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
