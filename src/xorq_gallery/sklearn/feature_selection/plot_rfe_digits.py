"""Recursive feature elimination
===============================

sklearn: Load digits dataset, apply RFE with LogisticRegression using MinMaxScaler
preprocessing to determine pixel importance rankings, visualize rankings as a heatmap
with numerical annotations.

xorq: Same RFE pipeline wrapped in Pipeline.from_instance, deferred fit to extract
feature rankings, deferred visualization of ranking heatmap.

Both produce identical ranking values.

Dataset: load_digits (sklearn handwritten digits 0-9)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import xorq.api as xo
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import MinMaxScaler
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
    """Load the digits dataset and return reshaped data, target, image shape."""
    digits = load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    y = digits.target
    image_shape = digits.images[0].shape
    return {"X": X, "y": y, "image_shape": image_shape}


# ---------------------------------------------------------------------------
# Pipeline construction (shared)
# ---------------------------------------------------------------------------


def _build_pipeline():
    """Build sklearn Pipeline with MinMaxScaler and RFE."""
    return SklearnPipeline(
        [
            ("scaler", MinMaxScaler()),
            ("rfe", RFE(estimator=LogisticRegression(), n_features_to_select=1, step=1)),
        ]
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_ranking(ranking, image_shape, title="Ranking of pixels with RFE\n(Logistic Regression)"):
    """Build ranking heatmap visualization.

    Parameters
    ----------
    ranking : ndarray
        2D ranking array (image shape)
    image_shape : tuple
        Shape of the original digit image
    title : str
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.matshow(ranking, cmap=plt.cm.Blues)

    # Add annotations for pixel numbers
    for i in range(ranking.shape[0]):
        for j in range(ranking.shape[1]):
            ax.text(j, i, str(int(ranking[i, j])), ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit, extract rankings
# =========================================================================


def sklearn_way(data):
    """Eager sklearn: fit RFE pipeline, extract pixel rankings."""
    X = data["X"]
    y = data["y"]
    image_shape = data["image_shape"]

    pipe = _build_pipeline()
    pipe.fit(X, y)
    ranking = pipe.named_steps["rfe"].ranking_.reshape(image_shape)

    return {"ranking": ranking}


# =========================================================================
# XORQ WAY -- deferred fit, deferred ranking extraction
# =========================================================================


def xorq_way(data):
    """Deferred xorq: wrap RFE pipeline in Pipeline.from_instance, fit deferred.

    Returns dict with fitted pipeline for extracting rankings in main().
    Nothing is executed until ``.execute()``.
    """
    import pandas as pd

    X = data["X"]
    y = data["y"]

    con = xo.connect()

    # Register dataset
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    table = con.register(df, "digits")

    # Feature columns
    features = tuple(f"f{i}" for i in range(X.shape[1]))

    # Wrap sklearn pipeline
    sklearn_pipe = _build_pipeline()
    xorq_pipe = Pipeline.from_instance(sklearn_pipe)

    # Fit deferred
    fitted = xorq_pipe.fit(table, features=features, target="target")

    return {"fitted": fitted, "table": table}


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    data = _load_data()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(data)

    print("\n=== XORQ WAY ===")
    xo_deferred = xorq_way(data)

    # Extract rankings from fitted pipeline
    fitted = xo_deferred["fitted"]
    rfe = fitted.fitted_steps[1].model
    xo_ranking = rfe.ranking_.reshape(data["image_shape"])

    # Assert numerical equivalence BEFORE plotting
    print("\n=== ASSERTIONS ===")
    print("Comparing RFE rankings (sklearn vs xorq):")
    print(f"  sklearn ranking shape: {sk_results['ranking'].shape}")
    print(f"  xorq ranking shape: {xo_ranking.shape}")
    print(f"  sklearn ranking min/max: {sk_results['ranking'].min()}/{sk_results['ranking'].max()}")
    print(f"  xorq ranking min/max: {xo_ranking.min()}/{xo_ranking.max()}")

    np.testing.assert_array_equal(sk_results['ranking'], xo_ranking)
    print("Assertions passed: sklearn and xorq rankings match exactly.")

    # Create deferred plot with extracted ranking
    def _build_ranking_plot(df):
        """Build ranking plot from deferred execution."""
        return _plot_ranking(xo_ranking, data["image_shape"])

    # Execute deferred plot
    print("\n=== PLOTTING ===")
    xo_png = deferred_matplotlib_plot(xo_deferred["table"], _build_ranking_plot).execute()

    # Build sklearn plot
    sk_fig = _plot_ranking(sk_results["ranking"], data["image_shape"])

    # Composite: sklearn (left) | xorq (right)
    xo_img = load_plot_bytes(xo_png)
    sk_img = fig_to_image(sk_fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(sk_img)
    axes[0].set_title("sklearn")
    axes[0].axis("off")

    axes[1].imshow(xo_img)
    axes[1].set_title("xorq")
    axes[1].axis("off")

    plt.suptitle("RFE Pixel Rankings on Digits: sklearn vs xorq", fontsize=14)
    plt.tight_layout()
    out = "imgs/plot_rfe_digits.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Composite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
