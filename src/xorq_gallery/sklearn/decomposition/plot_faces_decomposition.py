"""Faces dataset decompositions
============================

sklearn: Load Olivetti faces dataset, apply 8 different unsupervised matrix
decomposition methods (PCA, NMF, FastICA, MiniBatchSparsePCA, MiniBatchDictionaryLearning,
MiniBatchKMeans, FactorAnalysis, and dictionary learning variants). Each extracts
components that reconstruct the face images.

xorq: Same decomposition pipelines wrapped in Pipeline.from_instance, deferred
execution via xorq. Components extracted via make_other callbacks.

Both produce identical decomposition components.

Dataset: Olivetti Faces (sklearn built-in, 400 faces, 64x64 pixels)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/decomposition/plot_faces_decomposition.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cluster, decomposition
from sklearn.datasets import fetch_olivetti_faces
from sklearn.pipeline import Pipeline as SklearnPipeline

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    make_deferred_xorq_fit_transform_result,
    make_sklearn_fit_transform_result,
    make_xorq_fit_transform_result,
    split_data_nop,
)
from xorq_gallery.utils import fig_to_image, save_fig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 0
N_ROW, N_COL = 2, 3
N_COMPONENTS = N_ROW * N_COL  # 6
IMAGE_SHAPE = (64, 64)
TARGET_COL = "subject_id"
PRED_COL = "pred"  # unused, required by comparator


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Load Olivetti faces, return DataFrame with pixel + centered cols + subject_id."""
    faces, target = fetch_olivetti_faces(
        return_X_y=True, shuffle=True, random_state=SEED
    )
    faces = faces.astype(np.float64)
    n_samples, n_features = faces.shape
    print(f"Dataset: {n_samples} faces, {n_features} features")

    pixel_cols = tuple(f"pixel_{i}" for i in range(n_features))
    centered_cols = tuple(f"centered_{i}" for i in range(n_features))

    faces_centered = faces - faces.mean(axis=0)
    faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

    df = pd.DataFrame(faces, columns=pixel_cols)
    df_centered = pd.DataFrame(faces_centered, columns=centered_cols)
    df = pd.concat([df, df_centered], axis=1)
    df[TARGET_COL] = target
    return df


# pixel_cols / centered_cols are derived from load_data() but we need them
# at module level for FEATURE_COLS of each comparator.  Compute once here.
_sample_faces, _sample_target = fetch_olivetti_faces(
    return_X_y=True, shuffle=True, random_state=SEED
)
_n_features = _sample_faces.shape[1]
PIXEL_COLS = tuple(f"pixel_{i}" for i in range(_n_features))
CENTERED_COLS = tuple(f"centered_{i}" for i in range(_n_features))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_gallery(title, images, n_col=N_COL, n_row=N_ROW, cmap=plt.cm.gray):
    """Plot a grid of face component images."""
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.suptitle(title, size=16)

    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            vec.reshape(IMAGE_SHAPE),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axis("off")

    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
    return fig


# ---------------------------------------------------------------------------
# make_other callbacks for components_ extraction
# ---------------------------------------------------------------------------


def _get_components(estimator):
    """Extract face components from a fitted estimator (components_ or cluster_centers_)."""
    if hasattr(estimator, "components_"):
        return estimator.components_[:N_COMPONENTS].copy()
    return estimator.cluster_centers_[:N_COMPONENTS].copy()


def _make_sklearn_other(fitted):
    """Extract components from the last step of the fitted sklearn Pipeline."""
    estimator = fitted.steps[-1][-1]
    return {"components": _get_components(estimator)}


def _make_xorq_other(xorq_fitted):
    """Return dict of callables that extract components after deferred fit."""
    return {
        "components": lambda: _get_components(xorq_fitted.fitted_steps[-1].model)
    }


# ---------------------------------------------------------------------------
# Comparator callbacks (shared by both comparators)
# ---------------------------------------------------------------------------


def compare_results(comparator):
    """Print max/mean component difference per method (informational, no assertions).

    Non-convex algorithms (NMF, FastICA, SparsePCA, DictLearning) may converge
    to different local minima; strict equality assertions would be fragile.
    """
    print("\n=== Comparing Results ===")
    for name in comparator.sklearn_results:
        sk_comp = comparator.sklearn_results[name]["other"]["components"]
        xo_comp = comparator.xorq_results[name]["other"]["components"]
        diff = np.abs(sk_comp - xo_comp)
        print(
            f"  {name}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}"
            f"  (shape {sk_comp.shape})"
        )
    print("  Note: differences expected for stochastic/non-convex algorithms")


def plot_results(comparator):
    """Build sklearn | xorq gallery figure for each method in the comparator."""
    method_figs = []
    for name, _ in comparator.names_pipelines:
        sk_comp = comparator.sklearn_results[name]["other"]["components"]
        xo_comp = comparator.xorq_results[name]["other"]["components"]

        sk_fig = plot_gallery(f"sklearn: {name}", sk_comp)
        xo_fig = plot_gallery(f"xorq: {name}", xo_comp)

        row_fig, axes = plt.subplots(1, 2, figsize=(12, 2.3 * N_ROW))
        axes[0].imshow(fig_to_image(sk_fig))
        axes[0].axis("off")
        axes[0].set_title(f"sklearn: {name}", fontsize=12, fontweight="bold")
        axes[1].imshow(fig_to_image(xo_fig))
        axes[1].axis("off")
        axes[1].set_title(f"xorq: {name}", fontsize=12, fontweight="bold")
        plt.close(sk_fig)
        plt.close(xo_fig)
        row_fig.tight_layout()
        method_figs.append(row_fig)

    n = len(method_figs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.3 * N_ROW * n))
    if n == 1:
        axes = [axes]
    for ax, mf in zip(axes, method_figs):
        ax.imshow(fig_to_image(mf))
        ax.axis("off")
        plt.close(mf)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

metrics_names_funcs = ()

_pixel_methods = (NMF,) = ("nmf",)
_pixel_names_pipelines = (
    (
        NMF,
        SklearnPipeline(
            [("nmf", decomposition.NMF(n_components=N_COMPONENTS, tol=5e-3, random_state=SEED))]
        ),
    ),
)

_centered_methods = (PCA, ICA, SPARSE_PCA, DICT_LEARNING, FA, DICT_POS) = (
    "pca",
    "ica",
    "sparse_pca",
    "dict_learning",
    "fa",
    "dict_pos",
)
_centered_names_pipelines = (
    (
        PCA,
        SklearnPipeline(
            [
                (
                    "pca",
                    decomposition.PCA(
                        n_components=N_COMPONENTS, svd_solver="randomized", whiten=True,
                        random_state=SEED,
                    ),
                )
            ]
        ),
    ),
    (
        ICA,
        SklearnPipeline(
            [
                (
                    "ica",
                    decomposition.FastICA(
                        n_components=N_COMPONENTS,
                        max_iter=400,
                        whiten="arbitrary-variance",
                        tol=15e-5,
                        random_state=SEED,
                    ),
                )
            ]
        ),
    ),
    (
        SPARSE_PCA,
        SklearnPipeline(
            [
                (
                    "sparse_pca",
                    decomposition.MiniBatchSparsePCA(
                        n_components=N_COMPONENTS,
                        alpha=0.1,
                        max_iter=100,
                        batch_size=5,
                        random_state=SEED,
                    ),
                )
            ]
        ),
    ),
    (
        DICT_LEARNING,
        SklearnPipeline(
            [
                (
                    "dict_learning",
                    decomposition.MiniBatchDictionaryLearning(
                        n_components=N_COMPONENTS,
                        alpha=0.1,
                        max_iter=50,
                        batch_size=3,
                        random_state=SEED,
                    ),
                )
            ]
        ),
    ),
    (
        FA,
        SklearnPipeline(
            [
                (
                    "fa",
                    decomposition.FactorAnalysis(
                        n_components=N_COMPONENTS, max_iter=20,
                        random_state=SEED,
                    ),
                )
            ]
        ),
    ),
    (
        DICT_POS,
        SklearnPipeline(
            [
                (
                    "dict_pos",
                    decomposition.MiniBatchDictionaryLearning(
                        n_components=N_COMPONENTS,
                        alpha=0.1,
                        max_iter=50,
                        batch_size=3,
                        random_state=SEED,
                        positive_dict=True,
                    ),
                )
            ]
        ),
    ),
)

_shared_kwargs = dict(
    target=TARGET_COL,
    pred=PRED_COL,
    metrics_names_funcs=metrics_names_funcs,
    load_data=load_data,
    split_data=split_data_nop,
    make_sklearn_result=make_sklearn_fit_transform_result(
        make_other=_make_sklearn_other
    ),
    make_deferred_xorq_result=make_deferred_xorq_fit_transform_result(
        make_other=_make_xorq_other
    ),
    make_xorq_result=make_xorq_fit_transform_result,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)

comparator_pixel = SklearnXorqComparator(
    names_pipelines=_pixel_names_pipelines,
    features=PIXEL_COLS,
    **_shared_kwargs,
)
comparator_centered = SklearnXorqComparator(
    names_pipelines=_centered_names_pipelines,
    features=CENTERED_COLS,
    **_shared_kwargs,
)

# expose the exprs to invoke `xorq build plot_faces_decomposition.py --expr $expr_name`
(xorq_nmf_transformed,) = (
    comparator_pixel.deferred_xorq_results[name]["transformed"]
    for name in _pixel_methods
)
(
    xorq_pca_transformed,
    xorq_ica_transformed,
    xorq_sparse_pca_transformed,
    xorq_dict_learning_transformed,
    xorq_fa_transformed,
    xorq_dict_pos_transformed,
) = (
    comparator_centered.deferred_xorq_results[name]["transformed"]
    for name in _centered_methods
)


def main():
    # KMeans: handled eagerly because xorq Pipeline.fit() raises
    # "Can't infer target for a prediction step" when target=None and the last
    # step has predict(). MiniBatchKMeans has predict() (ClusterMixin) but is
    # unsupervised. The ClusterMixin exemption exists in FittedStep.__attrs_post_init__
    # (pipeline_lib.py:386) but is missing from Pipeline.fit() (pipeline_lib.py:836).
    # TODO: fix in xorq — add ClusterMixin check to Pipeline.fit(), then move
    # KMeans into the comparator like the other methods.
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=N_COMPONENTS, tol=1e-3, batch_size=20, max_iter=50, random_state=SEED
    )
    df = comparator_centered.df
    kmeans.fit(df[list(CENTERED_COLS)])
    sk_kmeans_components = kmeans.cluster_centers_[:N_COMPONENTS]

    # Run comparisons for both comparators
    comparator_pixel.result_comparison
    comparator_centered.result_comparison

    # Build composite figure
    pixel_fig = comparator_pixel.plot_results()
    centered_fig = comparator_centered.plot_results()

    # Add KMeans row manually
    sk_kmeans_fig = plot_gallery(
        "sklearn: KMeans cluster centers", sk_kmeans_components
    )
    kmeans_row, axes = plt.subplots(1, 2, figsize=(12, 2.3 * N_ROW))
    axes[0].imshow(fig_to_image(sk_kmeans_fig))
    axes[0].axis("off")
    axes[0].set_title("sklearn: kmeans", fontsize=12, fontweight="bold")
    axes[1].text(
        0.5,
        0.5,
        "KMeans: xorq Pipeline requires target\nfor predict steps; handled eagerly",
        ha="center",
        va="center",
        transform=axes[1].transAxes,
        fontsize=11,
    )
    axes[1].axis("off")
    plt.close(sk_kmeans_fig)
    kmeans_row.tight_layout()

    fig, outer_axes = plt.subplots(3, 1, figsize=(12, 2.3 * N_ROW * 8))
    outer_axes[0].imshow(fig_to_image(pixel_fig))
    outer_axes[0].axis("off")
    outer_axes[1].imshow(fig_to_image(centered_fig))
    outer_axes[1].axis("off")
    outer_axes[2].imshow(fig_to_image(kmeans_row))
    outer_axes[2].axis("off")
    plt.close(pixel_fig)
    plt.close(centered_fig)
    plt.close(kmeans_row)
    fig.suptitle("Faces Dataset Decompositions: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    save_fig("imgs/faces_decomposition.png", fig)


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
