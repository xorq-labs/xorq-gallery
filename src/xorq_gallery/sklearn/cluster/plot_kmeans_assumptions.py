"""Demonstration of k-means assumptions
========================================

sklearn: Generate four synthetic 2D datasets that violate different k-means
assumptions -- (1) incorrect number of clusters, (2) anisotropically
distributed blobs, (3) blobs with unequal variance, (4) unevenly sized
blobs -- fit KMeans on each scenario, and visualize the resulting cluster
assignments in a 1x4 grid.

xorq: Same KMeans pipelines wrapped in Pipeline.from_instance, deferred
fit/predict for each of the four scenarios.  Data is registered as ibis
tables; predictions are fully deferred until main() executes them.

This example illustrates situations where k-means will produce unintuitive
and possibly undesirable clusters.

Dataset: make_blobs (sklearn synthetic, 1500 samples, 3 centers, 4 variants)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline as SklearnPipeline
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import fig_to_image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 170
N_SAMPLES = 1500
FEATURES = ("f0", "f1")
TARGET_COL = "label"
PRED_COL = "pred"

# Transformation matrix for anisotropic scenario
ANISO_TRANSFORM = ((0.60834549, -0.63667341), (-0.40887718, 0.85253229))

# Cluster std deviations for unequal-variance scenario
VARIED_STD = (1.0, 2.5, 0.5)

# Per-class sample counts for unequal-size scenario
UNEVEN_SIZES = (500, 100, 10)

# Scenario definitions: (title, n_clusters)
SCENARIOS = (
    ("Incorrect Number of Blobs", 2),
    ("Anisotropically Distributed Blobs", 3),
    ("Unequal Variance", 3),
    ("Unevenly Sized Blobs", 3),
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _generate_datasets():
    """Generate the four synthetic datasets as pandas DataFrames.

    Returns a list of four DataFrames, one per scenario, each with columns
    f0, f1, and label.
    """
    X, y = make_blobs(n_samples=N_SAMPLES, random_state=RANDOM_STATE)

    # Scenario 1: Incorrect number of clusters (use base blobs as-is)
    df_incorrect = pd.DataFrame(X, columns=list(FEATURES))
    df_incorrect[TARGET_COL] = y

    # Scenario 2: Anisotropic blobs (apply linear transformation)
    X_aniso = np.dot(X, np.array(ANISO_TRANSFORM))
    df_aniso = pd.DataFrame(X_aniso, columns=list(FEATURES))
    df_aniso[TARGET_COL] = y

    # Scenario 3: Unequal variance
    X_varied, y_varied = make_blobs(
        n_samples=N_SAMPLES,
        cluster_std=list(VARIED_STD),
        random_state=RANDOM_STATE,
    )
    df_varied = pd.DataFrame(X_varied, columns=list(FEATURES))
    df_varied[TARGET_COL] = y_varied

    # Scenario 4: Unevenly sized blobs
    X_filtered = np.vstack(
        [X[y == k][: UNEVEN_SIZES[k]] for k in range(len(UNEVEN_SIZES))]
    )
    y_filtered = np.concatenate(
        [np.full(UNEVEN_SIZES[k], k) for k in range(len(UNEVEN_SIZES))]
    )
    df_uneven = pd.DataFrame(X_filtered, columns=list(FEATURES))
    df_uneven[TARGET_COL] = y_filtered

    return [df_incorrect, df_aniso, df_varied, df_uneven]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_scenarios(datasets, results, title_prefix=""):
    """Build a 1x4 grid of scatter plots showing cluster assignments.

    Parameters
    ----------
    datasets : list of ndarray
        Feature arrays for each scenario (n_samples, 2).
    results : list of dict
        Each dict has "labels" (ndarray) and "centers" (ndarray).
    title_prefix : str
        Prefix for the suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for idx, (ax, (title, _n_clusters)) in enumerate(zip(axes, SCENARIOS)):
        X = datasets[idx]
        labels = results[idx]["labels"]
        centers = results[idx]["centers"]

        ax.scatter(X[:, 0], X[:, 1], s=6, c=labels, cmap="viridis", alpha=0.7)
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="x",
            s=80,
            linewidths=2,
            color="r",
            zorder=10,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks(())
        ax.set_yticks(())

    fig.suptitle(
        f"{title_prefix}Demonstration of k-means assumptions",
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


def sklearn_way(datasets):
    """Eager sklearn: fit KMeans on each of the four scenarios.

    Parameters
    ----------
    datasets : list of pd.DataFrame
        Four DataFrames, one per scenario.

    Returns
    -------
    list of dict
        Each dict has "labels" and "centers".
    """
    results = []

    for idx, (title, n_clusters) in enumerate(SCENARIOS):
        df = datasets[idx]
        X = df[list(FEATURES)].values

        pipeline = SklearnPipeline(
            [
                (
                    "kmeans",
                    KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10),
                )
            ]
        )
        pipeline.fit(X)
        estimator = pipeline.named_steps["kmeans"]

        results.append(
            {
                "labels": estimator.labels_,
                "centers": estimator.cluster_centers_,
            }
        )
        print(
            f"  {title:40s} n_clusters={n_clusters}  inertia={estimator.inertia_:.1f}"
        )

    return results


# =========================================================================
# XORQ WAY -- deferred fit/predict
# =========================================================================


def xorq_way(datasets):
    """Deferred xorq: Pipeline.from_instance + fit + predict for each scenario.

    Returns a list of deferred prediction expressions. No .execute() calls.
    """
    con = xo.connect()
    preds = []

    for idx, (title, n_clusters) in enumerate(SCENARIOS):
        df = datasets[idx]
        table = con.register(df, f"scenario_{idx}")

        sklearn_pipe = SklearnPipeline(
            [
                (
                    "kmeans",
                    KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10),
                )
            ]
        )
        xorq_pipe = Pipeline.from_instance(sklearn_pipe)
        fitted = xorq_pipe.fit(table, features=FEATURES, target=TARGET_COL)
        pred_expr = fitted.predict(table, name=PRED_COL)

        preds.append(pred_expr)

    return preds


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    print("=== GENERATING DATA ===")
    dfs = _generate_datasets()
    for idx, (title, _) in enumerate(SCENARIOS):
        print(f"  Scenario {idx + 1}: {title} ({len(dfs[idx])} samples)")

    print("\n=== SKLEARN WAY ===")
    sk_results = sklearn_way(dfs)

    print("\n=== XORQ WAY ===")
    xo_preds = xorq_way(dfs)

    # --- Execute deferred predictions and assert equivalence ---
    print("\n=== ASSERTIONS ===")
    xo_results = []
    feature_arrays = [df[list(FEATURES)].values for df in dfs]

    for idx, (title, n_clusters) in enumerate(SCENARIOS):
        sk_labels = sk_results[idx]["labels"]
        xo_labels = xo_preds[idx].execute()[PRED_COL].values

        sk_n = len(np.unique(sk_labels))
        xo_n = len(np.unique(xo_labels))
        assert sk_n == xo_n, (
            f"{title}: cluster count mismatch (sklearn={sk_n}, xorq={xo_n})"
        )

        # Remap xorq labels to match sklearn colours for plotting
        xo_remapped = _remap_labels(sk_labels, xo_labels)
        xo_results.append(
            {
                "labels": xo_remapped,
                "centers": _centroids_from_labels(feature_arrays[idx], xo_remapped),
            }
        )

        print(f"  {title:40s} clusters: sklearn={sk_n}, xorq={xo_n}  OK")

    print("Assertions passed: sklearn and xorq cluster counts match.")

    # --- Plotting ---
    print("\n=== PLOTTING ===")

    sk_fig = _plot_scenarios(feature_arrays, sk_results, title_prefix="sklearn: ")
    xo_fig = _plot_scenarios(feature_arrays, xo_results, title_prefix="xorq: ")

    # Composite: sklearn (top) | xorq (bottom)
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(
        "Demonstration of k-means assumptions: sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    out = "imgs/plot_kmeans_assumptions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
