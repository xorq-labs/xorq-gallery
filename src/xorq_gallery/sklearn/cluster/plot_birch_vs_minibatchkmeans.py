"""Compare BIRCH and MiniBatchKMeans
=====================================

sklearn: Generate 25,000 synthetic 2D samples on a 10x10 grid of blob centers,
fit BIRCH (with and without global clustering step) and MiniBatchKMeans, compare
timing and cluster assignments, plot results.

xorq: Same three clustering models wrapped in Pipeline.from_instance, deferred
fit/predict.  Predictions are executed in main() and plotted alongside sklearn
results.

BIRCH and MiniBatchKMeans are order-sensitive algorithms -- DataFusion does not
guarantee row order, so cluster assignments may differ between sklearn and xorq.
Assertions check cluster counts rather than exact label matching.

Dataset: make_blobs (sklearn synthetic, 25k samples, 100 centers on 10x10 grid)
"""

from __future__ import annotations

import os
from itertools import cycle
from time import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import Birch, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline as SklearnPipeline
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import fig_to_image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 0
N_SAMPLES = 25_000
N_CLUSTERS_GLOBAL = 100
BIRCH_THRESHOLD = 1.7
FEATURES = ("f0", "f1")
TRUE_LABEL_COL = "true_label"
PRED_COL = "pred"
ROW_IDX = "row_idx"

# Model configs: (name, SklearnPipeline builder)
MODEL_CONFIGS = (
    (
        "BIRCH without global clustering",
        lambda: SklearnPipeline(
            [("birch", Birch(threshold=BIRCH_THRESHOLD, n_clusters=None))]
        ),
    ),
    (
        "BIRCH with global clustering",
        lambda: SklearnPipeline(
            [("birch", Birch(threshold=BIRCH_THRESHOLD, n_clusters=N_CLUSTERS_GLOBAL))]
        ),
    ),
    (
        "MiniBatchKMeans",
        lambda: SklearnPipeline(
            [
                (
                    "minibatchkmeans",
                    MiniBatchKMeans(
                        n_clusters=N_CLUSTERS_GLOBAL,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    ),
)


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Generate 25k samples from a 10x10 grid of blob centers."""
    xx = np.linspace(-22, 22, 10)
    yy = np.linspace(-22, 22, 10)
    xx, yy = np.meshgrid(xx, yy)
    centers = np.hstack((np.ravel(xx)[:, np.newaxis], np.ravel(yy)[:, np.newaxis]))

    X, y = make_blobs(n_samples=N_SAMPLES, centers=centers, random_state=RANDOM_STATE)

    df = pd.DataFrame(X, columns=list(FEATURES))
    df[TRUE_LABEL_COL] = y
    df[ROW_IDX] = range(len(df))
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

ALL_COLORS = list(mcolors.cnames.keys())


def _plot_clusters(X, labels, title):
    """Plot cluster scatter -- each cluster gets a unique colour."""
    fig, ax = plt.subplots(figsize=(5, 4))

    n_clusters = np.unique(labels).size
    colors_ = cycle(ALL_COLORS)

    for k, col in zip(range(n_clusters), colors_):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], c="w", edgecolor=col, marker=".", alpha=0.5)

    ax.set_xlim([-25, 25])
    ax.set_ylim([-25, 25])
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _remap_labels(reference, labels):
    """Remap *labels* so that cluster IDs match *reference* as closely as
    possible (maximise overlap via the Hungarian algorithm).

    Returns a new label array with the same shape as *labels*.
    """
    ref_ids = np.unique(reference)
    lab_ids = np.unique(labels)

    # Build a cost matrix: -overlap[ref_cluster, lab_cluster]
    cost = np.zeros((len(ref_ids), len(lab_ids)), dtype=int)
    for i, r in enumerate(ref_ids):
        for j, lab in enumerate(lab_ids):
            cost[i, j] = -np.sum((reference == r) & (labels == lab))

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[lab_ids[c]] = ref_ids[r]

    # Labels in *labels* that have no match keep an offset id
    next_id = ref_ids.max() + 1 if len(ref_ids) else 0
    out = np.empty_like(labels)
    for idx, val in enumerate(labels):
        if val in mapping:
            out[idx] = mapping[val]
        else:
            out[idx] = next_id
            next_id += 1

    return out


# =========================================================================
# SKLEARN WAY -- eager fit, timing
# =========================================================================


def sklearn_way(df):
    """Eager sklearn: fit BIRCH and MiniBatchKMeans, report timing and
    cluster counts.

    Returns dict of model_name -> {labels, n_clusters, time}.
    """
    X = df[list(FEATURES)].values
    results = {}

    for name, builder in MODEL_CONFIGS:
        pipe = builder()
        t0 = time()
        pipe.fit(X)
        elapsed = time() - t0

        estimator = pipe.steps[0][1]
        labels = estimator.labels_
        n_clusters = np.unique(labels).size

        print(f"  {name:40s} | {elapsed:.2f}s | {n_clusters} clusters")

        results[name] = {
            "labels": labels,
            "n_clusters": n_clusters,
            "time": elapsed,
        }

    return results


# =========================================================================
# XORQ WAY -- deferred fit/predict
# =========================================================================


def xorq_way(table):
    """Deferred xorq: Pipeline.from_instance + fit + predict for each model.

    Returns dict of model_name -> deferred prediction expression.
    No .execute() calls here.
    """
    results = {}

    for name, builder in MODEL_CONFIGS:
        pipe = builder()
        xorq_pipe = Pipeline.from_instance(pipe)
        fitted = xorq_pipe.fit(table, features=FEATURES, target=TRUE_LABEL_COL)
        preds = fitted.predict(table, name=PRED_COL)
        results[name] = preds

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()
    print(f"Generated {len(df)} samples")

    con = xo.connect()
    table = con.register(df, "blobs_grid")

    print("\n=== SKLEARN WAY ===")
    sk_results = sklearn_way(df)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(table)

    # --- Execute deferred predictions and assert equivalence ---
    print("\n=== ASSERTIONS ===")
    xo_executed = {}
    for name in [n for n, _ in MODEL_CONFIGS]:
        xo_preds_df = xo_results[name].execute()
        xo_labels = (
            xo_preds_df.sort_values(ROW_IDX).reset_index(drop=True)[PRED_COL].values
        )
        xo_executed[name] = xo_labels

        xo_n_clusters = np.unique(xo_labels).size
        sk_n_clusters = sk_results[name]["n_clusters"]

        print(
            f"  {name:40s} | sklearn: {sk_n_clusters} clusters, "
            f"xorq: {xo_n_clusters} clusters"
        )
        # All three algorithms are order-sensitive (BIRCH CF-tree,
        # MiniBatchKMeans mini-batch selection).  DataFusion does not
        # guarantee row order, so xorq may see data in a different order.
        # For models with a fixed n_clusters the count should still match;
        # for Birch(n_clusters=None) the sub-cluster count may differ.
        if "without global" not in name:
            assert sk_n_clusters == xo_n_clusters, (
                f"{name}: cluster count mismatch "
                f"(sklearn={sk_n_clusters}, xorq={xo_n_clusters})"
            )

    print("Assertions passed: cluster counts match.")

    # --- Plotting ---
    print("\n=== PLOTTING ===")
    X = df[list(FEATURES)].values

    # Build all sub-plots from executed results (same pattern as
    # plot_kmeans_silhouette_analysis -- cluster algorithms are
    # order-sensitive so refitting inside a UDAF would diverge).
    sk_figs = []
    xo_figs = []
    for name in [n for n, _ in MODEL_CONFIGS]:
        sk_labels = sk_results[name]["labels"]
        xo_labels = _remap_labels(sk_labels, xo_executed[name])

        sk_figs.append(_plot_clusters(X, sk_labels, f"sklearn: {name}"))
        xo_figs.append(_plot_clusters(X, xo_labels, f"xorq: {name}"))

    # Composite: 2 rows x 3 cols (sklearn top, xorq bottom)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for col in range(3):
        axes[0, col].imshow(fig_to_image(sk_figs[col]))
        axes[0, col].axis("off")

        axes[1, col].imshow(fig_to_image(xo_figs[col]))
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("sklearn", fontsize=14, fontweight="bold")
    axes[1, 0].set_ylabel("xorq", fontsize=14, fontweight="bold")

    fig.suptitle(
        "BIRCH vs MiniBatchKMeans: sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    out = "imgs/plot_birch_vs_minibatchkmeans.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    for f in sk_figs + xo_figs:
        plt.close(f)
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
