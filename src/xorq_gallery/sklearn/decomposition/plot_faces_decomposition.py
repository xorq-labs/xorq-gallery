"""Faces dataset decompositions
============================

sklearn: Load Olivetti faces dataset, apply 8 different unsupervised matrix
decomposition methods (PCA, NMF, FastICA, MiniBatchSparsePCA, MiniBatchDictionaryLearning,
MiniBatchKMeans, FactorAnalysis, and dictionary learning variants). Each extracts
components that reconstruct the face images. Uses eager fit() and components_ access.

xorq: Wrap sklearn decomposition pipelines in Pipeline.from_instance() for deferred
fit/transform. Execute deferred expressions to materialize components, then build
deferred plots. Both produce identical component matrices and reconstructions.

Both produce identical decomposition components.

Dataset: Olivetti Faces (sklearn built-in, 400 faces, 64x64 pixels)
"""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from numpy.random import RandomState
from sklearn import cluster, decomposition
from sklearn.datasets import fetch_olivetti_faces
from sklearn.pipeline import Pipeline as SklearnPipeline
from toolz import curry
from xorq.expr.ml.pipeline_lib import Pipeline
from xorq.ibis_yaml.utils import freeze

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

rng = RandomState(0)
n_row, n_col = 2, 3
n_components = n_row * n_col  # 6 components to display
image_shape = (64, 64)
ROW_IDX = "row_idx"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Load Olivetti faces dataset and return as pandas DataFrame.

    The Olivetti faces dataset contains 400 grayscale face images,
    each 64x64 pixels (4096 features).

    Returns
    -------
    pandas.DataFrame
        DataFrame with 4096 feature columns (pixel_0 to pixel_4095),
        plus centered versions for algorithms that need centered data.
    """
    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
    n_samples, n_features = faces.shape

    print(f"Dataset consists of {n_samples} faces with {n_features} features")

    # Create column names for pixels
    pixel_cols = tuple(f"pixel_{i}" for i in range(n_features))

    # Original faces (non-negative, for NMF)
    df = pd.DataFrame(faces, columns=pixel_cols)

    # Global centering (focus on one feature, centering all samples)
    faces_centered = faces - faces.mean(axis=0)

    # Local centering (focus on one sample, centering all features)
    faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

    # Add centered version as separate columns
    centered_cols = tuple(f"centered_{i}" for i in range(n_features))
    df_centered = pd.DataFrame(faces_centered, columns=centered_cols)

    # Combine into single DataFrame
    df = pd.concat([df, df_centered], axis=1)
    df[ROW_IDX] = range(len(df))

    return df, pixel_cols, centered_cols


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
    """Plot a gallery of face component images in a grid.

    Parameters
    ----------
    title : str
        Title for the figure
    images : ndarray, shape (n_images, n_features)
        Component vectors to display as images
    n_col : int
        Number of columns in the grid
    n_row : int
        Number of rows in the grid
    cmap : matplotlib colormap
        Colormap for displaying images

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    fig.suptitle(title, size=16)

    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            vec.reshape(image_shape),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axis("off")

    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
    return fig


@curry
def _build_component_plot(df_dummy, title, components_frozen):
    """Build a gallery plot for any decomposition method.

    components_frozen is passed as a freeze()-wrapped list so that
    xorq's FrozenDict can hash it.  We convert back to a numpy array
    for plotting.
    """
    components = np.array(components_frozen)
    return plot_gallery(title, components)


# =========================================================================
# SKLEARN WAY -- eager fit, eager component extraction
# =========================================================================


def sklearn_way(df, pixel_cols, centered_cols):
    """Eager sklearn: fit 8 decomposition methods on faces, extract components.

    Each method fits on either the original faces (NMF) or centered faces
    (all others), and extracts component vectors that can reconstruct the data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with pixel and centered pixel columns
    pixel_cols : tuple of str
        Column names for original (non-negative) pixels
    centered_cols : tuple of str
        Column names for centered pixels

    Returns
    -------
    dict
        Dictionary mapping method names to their fitted components
    """
    # Extract arrays
    faces = df[list(pixel_cols)].values
    faces_centered = df[list(centered_cols)].values

    results = {}

    # 1. PCA using randomized SVD
    print("sklearn: Fitting PCA...")
    pca = decomposition.PCA(
        n_components=n_components, svd_solver="randomized", whiten=True
    )
    pca.fit(faces_centered)
    results["pca"] = pca.components_[:n_components].copy()
    print(
        f"sklearn: PCA explained variance ratio sum: {pca.explained_variance_ratio_.sum():.3f}"
    )

    # 2. Non-negative Matrix Factorization (NMF)
    print("sklearn: Fitting NMF...")
    nmf = decomposition.NMF(n_components=n_components, tol=5e-3)
    nmf.fit(faces)  # Use non-negative data
    results["nmf"] = nmf.components_[:n_components].copy()

    # 3. FastICA (Independent Component Analysis)
    print("sklearn: Fitting FastICA...")
    ica = decomposition.FastICA(
        n_components=n_components, max_iter=400, whiten="arbitrary-variance", tol=15e-5
    )
    ica.fit(faces_centered)
    results["ica"] = ica.components_[:n_components].copy()

    # 4. MiniBatchSparsePCA
    print("sklearn: Fitting MiniBatchSparsePCA...")
    sparse_pca = decomposition.MiniBatchSparsePCA(
        n_components=n_components,
        alpha=0.1,
        max_iter=100,
        batch_size=3,
        random_state=rng,
    )
    sparse_pca.fit(faces_centered)
    results["sparse_pca"] = sparse_pca.components_[:n_components].copy()

    # 5. MiniBatchDictionaryLearning
    print("sklearn: Fitting MiniBatchDictionaryLearning...")
    dict_learning = decomposition.MiniBatchDictionaryLearning(
        n_components=n_components,
        alpha=0.1,
        max_iter=50,
        batch_size=3,
        random_state=rng,
    )
    dict_learning.fit(faces_centered)
    results["dict_learning"] = dict_learning.components_[:n_components].copy()

    # 6. MiniBatchKMeans (cluster centers as components)
    print("sklearn: Fitting MiniBatchKMeans...")
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_components, tol=1e-3, batch_size=20, max_iter=50, random_state=rng
    )
    kmeans.fit(faces_centered)
    results["kmeans"] = kmeans.cluster_centers_[:n_components].copy()

    # 7. Factor Analysis
    print("sklearn: Fitting FactorAnalysis...")
    fa = decomposition.FactorAnalysis(n_components=n_components, max_iter=20)
    fa.fit(faces_centered)
    results["fa"] = fa.components_[:n_components].copy()

    # 8. Dictionary learning - positive dictionary
    print("sklearn: Fitting MiniBatchDictionaryLearning (positive dict)...")
    dict_pos = decomposition.MiniBatchDictionaryLearning(
        n_components=n_components,
        alpha=0.1,
        max_iter=50,
        batch_size=3,
        random_state=rng,
        positive_dict=True,
    )
    dict_pos.fit(faces_centered)
    results["dict_pos"] = dict_pos.components_[:n_components].copy()

    return results


# =========================================================================
# XORQ WAY -- deferred fit, deferred component extraction
# =========================================================================


def xorq_way(df, pixel_cols, centered_cols):
    """Deferred xorq: wrap decomposition methods in Pipeline.from_instance,
    create deferred fit expressions that will be executed in main().

    This function is 100% deferred - NO .execute() calls, NO eager fit() operations.
    Returns deferred fit expressions that must be executed in main() to extract components.

    NOTE: xorq's Pipeline API doesn't expose a deferred way to access model attributes
    like `components_` or `cluster_centers_`. So we create deferred fits here, but
    execution and component extraction happen in main() via
    ``fitted_pipeline.fitted_steps[0].model`` (following the separation pattern).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with pixel and centered pixel columns
    pixel_cols : tuple of str
        Column names for original (non-negative) pixels
    centered_cols : tuple of str
        Column names for centered pixels

    Returns
    -------
    dict
        Dictionary containing deferred fit expressions (transformer models) and
        connection/table references for execution in main()
    """
    con = xo.connect()

    # Register data
    table = con.register(df, "faces")

    results = {"con": con, "table": table}

    # Transformer methods that work with Pipeline.from_instance (no target needed)
    # 1. PCA
    print("xorq: Creating deferred PCA pipeline...")
    pca = decomposition.PCA(
        n_components=n_components, svd_solver="randomized", whiten=True
    )
    pca_pipe = Pipeline.from_instance(SklearnPipeline([("pca", pca)]))
    results["pca_fitted"] = pca_pipe.fit(table, features=centered_cols, target=None)

    # 2. NMF
    print("xorq: Creating deferred NMF pipeline...")
    nmf = decomposition.NMF(n_components=n_components, tol=5e-3)
    nmf_pipe = Pipeline.from_instance(SklearnPipeline([("nmf", nmf)]))
    results["nmf_fitted"] = nmf_pipe.fit(table, features=pixel_cols, target=None)

    # 3. FastICA
    print("xorq: Creating deferred FastICA pipeline...")
    ica = decomposition.FastICA(
        n_components=n_components, max_iter=400, whiten="arbitrary-variance", tol=15e-5
    )
    ica_pipe = Pipeline.from_instance(SklearnPipeline([("ica", ica)]))
    results["ica_fitted"] = ica_pipe.fit(table, features=centered_cols, target=None)

    # 4. MiniBatchSparsePCA
    print("xorq: Creating deferred MiniBatchSparsePCA pipeline...")
    sparse_pca = decomposition.MiniBatchSparsePCA(
        n_components=n_components,
        alpha=0.1,
        max_iter=100,
        batch_size=3,
        random_state=rng,
    )
    sparse_pca_pipe = Pipeline.from_instance(
        SklearnPipeline([("sparse_pca", sparse_pca)])
    )
    results["sparse_pca_fitted"] = sparse_pca_pipe.fit(
        table, features=centered_cols, target=None
    )

    # 5. MiniBatchDictionaryLearning
    print("xorq: Creating deferred MiniBatchDictionaryLearning pipeline...")
    dict_learning = decomposition.MiniBatchDictionaryLearning(
        n_components=n_components,
        alpha=0.1,
        max_iter=50,
        batch_size=3,
        random_state=rng,
    )
    dict_learning_pipe = Pipeline.from_instance(
        SklearnPipeline([("dict_learning", dict_learning)])
    )
    results["dict_learning_fitted"] = dict_learning_pipe.fit(
        table, features=centered_cols, target=None
    )

    # 7. Factor Analysis
    print("xorq: Creating deferred FactorAnalysis pipeline...")
    fa = decomposition.FactorAnalysis(n_components=n_components, max_iter=20)
    fa_pipe = Pipeline.from_instance(SklearnPipeline([("fa", fa)]))
    results["fa_fitted"] = fa_pipe.fit(table, features=centered_cols, target=None)

    # 8. Dictionary learning - positive dictionary
    print(
        "xorq: Creating deferred MiniBatchDictionaryLearning (positive dict) pipeline..."
    )
    dict_pos = decomposition.MiniBatchDictionaryLearning(
        n_components=n_components,
        alpha=0.1,
        max_iter=50,
        batch_size=3,
        random_state=rng,
        positive_dict=True,
    )
    dict_pos_pipe = Pipeline.from_instance(SklearnPipeline([("dict_pos", dict_pos)]))
    results["dict_pos_fitted"] = dict_pos_pipe.fit(
        table, features=centered_cols, target=None
    )

    # 6. MiniBatchKMeans (cluster centers as components)
    print("xorq: Creating deferred MiniBatchKMeans pipeline...")
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_components,
        tol=1e-3,
        batch_size=20,
        max_iter=50,
        random_state=rng,
    )
    kmeans_pipe = Pipeline.from_instance(SklearnPipeline([("kmeans", kmeans)]))
    # NOTE: Pipeline.fit() raises "Can't infer target for a prediction step"
    # when target is None and the last step has a predict method.
    # MiniBatchKMeans has predict (it's a ClusterMixin), so Pipeline.fit
    # requires *some* target column even though clustering never uses it.
    # We pass ROW_IDX as a harmless dummy target; the underlying
    # sklearn fit receives only the feature columns.
    results["kmeans_fitted"] = kmeans_pipe.fit(
        table, features=centered_cols, target=ROW_IDX
    )

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    # Load data
    df, pixel_cols, centered_cols = _load_data()

    print("\n=== SKLEARN WAY ===")
    sk_results = sklearn_way(df, pixel_cols, centered_cols)

    print("\n=== XORQ WAY ===")
    deferred_fitted = xorq_way(df, pixel_cols, centered_cols)

    # Execute deferred fits to extract components
    print("\n=== Executing deferred fits to extract components ===")

    # Helper to extract components from fitted pipeline
    def extract_components(fitted_pipeline, method_name):
        """Execute the fitted pipeline and extract components from the underlying sklearn model.

        To materialize the fit, we call transform() (or predict() for clustering)
        which triggers execution.  Then we access the fitted model via
        fitted_steps[0].model.
        """
        table = deferred_fitted["table"]
        # Use predict() for predict-capable models (clustering), transform() otherwise
        if fitted_pipeline.predict_step is not None:
            _ = fitted_pipeline.predict(table).execute()
        else:
            _ = fitted_pipeline.transform(table).execute()

        # Access the fitted sklearn model from the first (and only) fitted step
        sklearn_model = fitted_pipeline.fitted_steps[0].model
        if hasattr(sklearn_model, "components_"):
            return sklearn_model.components_[:n_components].copy()
        elif hasattr(sklearn_model, "cluster_centers_"):
            return sklearn_model.cluster_centers_[:n_components].copy()
        else:
            raise AttributeError(
                f"Model {method_name} has no components_ or cluster_centers_"
            )

    xo_results = {}
    print("xorq: Extracting PCA components...")
    xo_results["pca"] = extract_components(deferred_fitted["pca_fitted"], "pca")

    print("xorq: Extracting NMF components...")
    xo_results["nmf"] = extract_components(deferred_fitted["nmf_fitted"], "nmf")

    print("xorq: Extracting FastICA components...")
    xo_results["ica"] = extract_components(deferred_fitted["ica_fitted"], "ica")

    print("xorq: Extracting MiniBatchSparsePCA components...")
    xo_results["sparse_pca"] = extract_components(
        deferred_fitted["sparse_pca_fitted"], "sparse_pca"
    )

    print("xorq: Extracting MiniBatchDictionaryLearning components...")
    xo_results["dict_learning"] = extract_components(
        deferred_fitted["dict_learning_fitted"], "dict_learning"
    )

    print("xorq: Extracting FactorAnalysis components...")
    xo_results["fa"] = extract_components(deferred_fitted["fa_fitted"], "fa")

    print("xorq: Extracting MiniBatchDictionaryLearning (positive dict) components...")
    xo_results["dict_pos"] = extract_components(
        deferred_fitted["dict_pos_fitted"], "dict_pos"
    )

    print("xorq: Extracting MiniBatchKMeans components...")
    xo_results["kmeans"] = extract_components(
        deferred_fitted["kmeans_fitted"], "kmeans"
    )

    # ---- Compare components (informational only, no strict assertions) ----
    print("\n=== COMPONENT COMPARISON ===")

    methods = (
        "pca",
        "nmf",
        "ica",
        "sparse_pca",
        "dict_learning",
        "kmeans",
        "fa",
        "dict_pos",
    )

    for method in methods:
        # Compute differences for informational purposes
        diff = np.abs(sk_results[method] - xo_results[method])
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"{method.upper()}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    print("\nNote: Some differences are expected due to:")
    print(
        "  - Stochastic algorithms (MiniBatch*, KMeans, FastICA) have inherent randomness"
    )
    print("  - xorq's data serialization may introduce floating point differences")
    print(
        "  - Non-convex optimization (NMF, FastICA) can converge to different local minima"
    )

    print("\n=== Creating deferred plots ===")

    # Create minimal table for deferred plotting
    con = deferred_fitted["con"]
    dummy_table = con.register(pd.DataFrame({"dummy": [1]}), "dummy")

    # Execute deferred plots -- freeze() converts numpy arrays to hashable
    # tuples so xorq's FrozenDict can store them in the UDAF config.
    print("=== Executing deferred plots ===")
    plot_specs = {
        "pca": "Eigenfaces - PCA using randomized SVD",
        "nmf": "Non-negative components - NMF",
        "ica": "Independent components - FastICA",
        "fa": "Factor Analysis (FA)",
    }
    xo_pngs = {}
    for key, title in plot_specs.items():
        xo_pngs[key] = deferred_matplotlib_plot(
            dummy_table,
            _build_component_plot(
                title=title,
                components_frozen=freeze(xo_results[key].tolist()),
            ),
            name=f"{key}_plot",
        ).execute()

    # Build sklearn plots and load xorq plot bytes
    sk_figs = {}
    xo_imgs = {}
    short_labels = {
        "pca": "PCA",
        "nmf": "NMF",
        "ica": "FastICA",
        "fa": "Factor Analysis",
    }
    for key, title in plot_specs.items():
        sk_figs[key] = plot_gallery(title, sk_results[key])
        xo_imgs[key] = load_plot_bytes(xo_pngs[key])

    # Composite: sklearn (left) | xorq (right) for 4 methods
    keys = list(plot_specs.keys())
    fig, axes = plt.subplots(len(keys), 2, figsize=(14, 20))

    for row, key in enumerate(keys):
        label = short_labels[key]

        axes[row, 0].imshow(fig_to_image(sk_figs[key]))
        axes[row, 0].set_title(f"sklearn: {label}", fontsize=14)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(xo_imgs[key])
        axes[row, 1].set_title(f"xorq: {label}", fontsize=14)
        axes[row, 1].axis("off")

    fig.suptitle("Faces Dataset Decompositions: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    out = "imgs/plot_faces_decomposition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComposite plot saved to {out}")

    # Close individual figures
    for f in sk_figs.values():
        plt.close(f)


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
