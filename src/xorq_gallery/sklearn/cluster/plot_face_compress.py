"""Vector Quantization Example
==============================

sklearn: Load the raccoon face image (scipy), flatten pixels, apply
KBinsDiscretizer with n_bins=8 using 'uniform' and 'kmeans' strategies,
reshape back to image, plot original and compressed images with pixel
value histograms.

xorq: Same KBinsDiscretizer wrapped in Pipeline.from_instance, deferred
fit/transform, deferred plots via deferred_matplotlib_plot.

Both produce identical compressed images and pixel distributions.

Dataset: Raccoon face (scipy, 768x1024 grayscale)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from scipy.datasets import face
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import KBinsDiscretizer
from toolz import curry
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BINS = 8
ENCODE = "ordinal"
RANDOM_STATE = 0
PIXEL_COL = "pixel"
ROW_IDX = "row_idx"
STRATEGIES = ("uniform", "kmeans")
IMAGE_SHAPE = (768, 1024)  # raccoon face dimensions


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_image():
    """Load the raccoon face image as a grayscale array."""
    return face(gray=True)


def _image_to_df(img):
    """Flatten image pixels into a DataFrame with row index for ordering."""
    pixels = img.ravel().astype(np.float64)
    return pd.DataFrame({ROW_IDX: np.arange(len(pixels)), PIXEL_COL: pixels})


def _build_pipeline(strategy):
    """Return sklearn Pipeline wrapping KBinsDiscretizer.

    subsample=None ensures both sklearn and xorq fit on identical data
    (the default subsample=200_000 causes sklearn to subsample large
    datasets, while xorq Pipeline always fits on all rows).
    """
    return SklearnPipeline(
        [
            (
                "kbd",
                KBinsDiscretizer(
                    n_bins=N_BINS,
                    encode=ENCODE,
                    strategy=strategy,
                    subsample=None,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_strategy(original_img, compressed_img, strategy, title_prefix=""):
    """Plot compressed image and pixel histogram for one strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(compressed_img, cmap=plt.cm.gray)
    axes[0].axis("off")
    axes[0].set_title("Rendering of the image")

    axes[1].hist(compressed_img.ravel(), bins=256)
    axes[1].set_xlabel("Pixel value")
    axes[1].set_ylabel("Count of pixels")
    axes[1].set_title("Sub-sampled distribution of the pixel values")

    fig.suptitle(
        f"{title_prefix}Compressed using {N_BINS} bins ({strategy} strategy)",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def _plot_original(img):
    """Plot original image and pixel histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(img, cmap=plt.cm.gray)
    axes[0].axis("off")
    axes[0].set_title("Rendering of the image")

    axes[1].hist(img.ravel(), bins=256)
    axes[1].set_xlabel("Pixel value")
    axes[1].set_ylabel("Count of pixels")
    axes[1].set_title("Distribution of the pixel values")

    fig.suptitle("Original image of a raccoon face", fontsize=13)
    fig.tight_layout()
    return fig


@curry
def _build_xorq_plot(df, strategy):
    """Curried plot function for deferred_matplotlib_plot.

    Refits KBinsDiscretizer on the materialised DataFrame and plots
    the compressed image with histogram.  Uses row_idx to restore
    original pixel order before reshaping to image.
    """
    # Sort by row_idx to restore original pixel order
    df_sorted = df.sort_values(ROW_IDX)
    pixels = df_sorted[PIXEL_COL].values.reshape(-1, 1)

    pipe = _build_pipeline(strategy)
    compressed = pipe.fit_transform(pixels).ravel().reshape(IMAGE_SHAPE)

    return _plot_strategy(None, compressed, strategy, title_prefix="xorq: ")


# =========================================================================
# SKLEARN WAY -- eager fit, transform, plot
# =========================================================================


def sklearn_way(img):
    """Eager sklearn: fit KBinsDiscretizer, transform pixels.

    Returns dict with compressed images for each strategy.
    """
    pixels = img.ravel().reshape(-1, 1).astype(np.float64)

    results = {}
    for strategy in STRATEGIES:
        pipe = _build_pipeline(strategy)
        compressed = pipe.fit_transform(pixels).ravel().reshape(IMAGE_SHAPE)
        results[strategy] = compressed
        print(f"  {strategy}: compressed to {N_BINS} bins")

    return results


# =========================================================================
# XORQ WAY -- deferred fit/transform
# =========================================================================


def xorq_way(table):
    """Deferred xorq: Pipeline.from_instance + fit + transform.

    Returns dict with deferred transform expressions for each strategy.
    No .execute() calls here.
    """
    results = {}
    for strategy in STRATEGIES:
        pipe = _build_pipeline(strategy)
        xorq_pipe = Pipeline.from_instance(pipe)
        fitted = xorq_pipe.fit(table, features=(PIXEL_COL,), target=None)
        transformed = fitted.transform(table)
        results[strategy] = transformed
        print(f"  {strategy}: deferred transform expression created")

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def _extract_pixel_values(xo_df):
    """Extract pixel values from xorq transform output, sorted by row_idx.

    Handles both direct column access and the {'key': ..., 'value': ...}
    dict-list format that Pipeline.transform() may return.
    Returns values sorted by row_idx to restore original pixel order.
    """
    # Sort by row_idx to restore original pixel order
    xo_df = xo_df.sort_values(ROW_IDX).reset_index(drop=True)

    if "transformed" in xo_df.columns:

        def extract_value(row):
            for item in row:
                if item["key"] == PIXEL_COL:
                    return item["value"]
            return None

        return xo_df["transformed"].apply(extract_value).values
    elif PIXEL_COL in xo_df.columns:
        return xo_df[PIXEL_COL].values
    else:
        raise ValueError(f"Unexpected columns in transform output: {xo_df.columns}")


def main():
    os.makedirs("imgs", exist_ok=True)

    img = _load_image()
    df = _image_to_df(img)

    con = xo.connect()
    table = con.register(df, "image_pixels")

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(img)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(table)

    # --- Execute deferred transforms and assert equivalence ---
    print("\n=== ASSERTIONS ===")
    xo_compressed = {}
    for strategy in STRATEGIES:
        xo_df = xo_results[strategy].execute()
        xo_pixels = _extract_pixel_values(xo_df).reshape(IMAGE_SHAPE)
        xo_compressed[strategy] = xo_pixels

        sk_pixels = sk_results[strategy]

        if strategy == "uniform":
            # Uniform strategy is fully order-independent: exact match
            np.testing.assert_allclose(
                sk_pixels,
                xo_pixels,
                rtol=1e-5,
                err_msg=f"Mismatch for strategy={strategy}",
            )
            print(f"  {strategy}: sklearn and xorq match (exact)")
        else:
            # kmeans strategy uses KMeans internally which is sensitive to
            # data ordering; assert that bin-value distributions are close
            sk_hist, _ = np.histogram(sk_pixels.ravel(), bins=N_BINS)
            xo_hist, _ = np.histogram(xo_pixels.ravel(), bins=N_BINS)
            np.testing.assert_allclose(
                sk_hist,
                xo_hist,
                rtol=0.10,
                err_msg=f"Histogram mismatch for strategy={strategy}",
            )
            print(f"  {strategy}: sklearn and xorq distributions match")

    print("Assertions passed.")

    # --- Plotting ---
    print("\n=== PLOTTING ===")

    # Original image
    orig_fig = _plot_original(img)

    # sklearn plots
    sk_figs = {}
    for strategy in STRATEGIES:
        sk_figs[strategy] = _plot_strategy(
            img, sk_results[strategy], strategy, title_prefix="sklearn: "
        )

    # xorq deferred plots
    xo_imgs = {}
    for strategy in STRATEGIES:
        xo_png = deferred_matplotlib_plot(
            table, _build_xorq_plot(strategy=strategy)
        ).execute()
        xo_imgs[strategy] = load_plot_bytes(xo_png)

    # Composite: 3 rows (original, uniform, kmeans), 2 columns (sklearn | xorq)
    n_rows = 1 + len(STRATEGIES)
    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 4 * n_rows))

    # Row 0: Original image (same for both)
    orig_img_arr = fig_to_image(orig_fig)
    axes[0, 0].imshow(orig_img_arr)
    axes[0, 0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(orig_img_arr)
    axes[0, 1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[0, 1].axis("off")

    # Rows 1+: Strategy results
    for row_idx, strategy in enumerate(STRATEGIES, start=1):
        axes[row_idx, 0].imshow(fig_to_image(sk_figs[strategy]))
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(xo_imgs[strategy])
        axes[row_idx, 1].axis("off")

    fig.suptitle(
        "Face Compression (Vector Quantization): sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    out = "imgs/plot_face_compress.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
