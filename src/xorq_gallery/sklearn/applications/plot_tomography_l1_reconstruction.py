"""Compressive sensing: tomography reconstruction with L1 prior (Lasso)
=======================================================================

sklearn: Build a sparse projection operator simulating a CT scanner,
generate noisy projections, solve the inverse problem with Ridge (L2) and
Lasso (L1) directly on numpy arrays. Compare reconstructions visually.

xorq: Register the projection matrix as an ibis table, fit Ridge and
Lasso via Pipeline.from_instance. Same results, deferred execution.

Dataset: Synthetic binary image (128x128)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/applications/plot_tomography_l1_reconstruction.py
"""

from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, sparse
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline as SklearnPipeline

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    split_data_nop,
)
from xorq_gallery.utils import (
    fig_to_image,
    save_fig,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE = 128
IMAGE_SHAPE = (IMG_SIZE, IMG_SIZE)
N_PIXELS = IMG_SIZE * IMG_SIZE
TARGET_COL = "projection"
PRED_COL = "pred"
FEATURE_COLS = tuple(f"px_{i}" for i in range(N_PIXELS))

np.random.seed(0)


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------


def _weights(x, dx=1, orig=0):
    x = np.ravel(x)
    floor_x = np.floor((x - orig) / dx).astype(np.int64)
    alpha = (x - orig - floor_x * dx) / dx
    return np.hstack((floor_x, floor_x + 1)), np.hstack((1 - alpha, alpha))


def _generate_center_coordinates(l_x):
    X, Y = np.mgrid[:l_x, :l_x].astype(np.float64)
    center = l_x / 2.0
    X += 0.5 - center
    Y += 0.5 - center
    return X, Y


def build_projection_operator(l_x, n_dir):
    X, Y = _generate_center_coordinates(l_x)
    angles = np.linspace(0, np.pi, n_dir, endpoint=False)
    data_inds, weights, camera_inds = [], [], []
    data_unravel_indices = np.arange(l_x**2)
    data_unravel_indices = np.hstack((data_unravel_indices, data_unravel_indices))
    for i, angle in enumerate(angles):
        Xrot = np.cos(angle) * X - np.sin(angle) * Y
        inds, w = _weights(Xrot, dx=1, orig=X.min())
        mask = np.logical_and(inds >= 0, inds < l_x)
        weights += list(w[mask])
        camera_inds += list(inds[mask] + i * l_x)
        data_inds += list(data_unravel_indices[mask])
    return sparse.coo_matrix((weights, (camera_inds, data_inds))).tocsr()


def generate_synthetic_data(l_x):
    rs = np.random.RandomState(0)
    n_pts = 36
    x, y = np.ogrid[0:l_x, 0:l_x]
    mask_outer = (x - l_x / 2.0) ** 2 + (y - l_x / 2.0) ** 2 < (l_x / 2.0) ** 2
    mask = np.zeros((l_x, l_x))
    points = l_x * rs.rand(2, n_pts)
    mask[points[0].astype(int), points[1].astype(int)] = 1
    mask = ndimage.gaussian_filter(mask, sigma=l_x / n_pts)
    res = np.logical_and(mask > mask.mean(), mask_outer)
    return np.logical_xor(res, ndimage.binary_erosion(res)).astype(float)


@cache
def _generate_raw_data():
    n_angles = IMG_SIZE // 7
    proj_operator = build_projection_operator(IMG_SIZE, n_angles)
    image = generate_synthetic_data(IMG_SIZE)
    proj = proj_operator @ image.ravel()[:, np.newaxis]
    proj += 0.15 * np.random.randn(*proj.shape)
    proj_flat = proj.ravel()
    return proj_operator, image, proj_flat, n_angles


def load_data():
    """Return projection matrix as flat DataFrame (rows=rays, cols=px_0..px_N + target)."""
    proj_operator, _image, proj_flat, _n_angles = _generate_raw_data()
    proj_dense = proj_operator.toarray()
    df = pd.DataFrame(proj_dense, columns=FEATURE_COLS)
    df[TARGET_COL] = proj_flat
    return df


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def compare_results(comparator):
    _, image, _, _ = _generate_raw_data()
    image_flat = image.ravel()

    print("\n=== Reconstruction Errors ===")
    for name, sklearn_result in comparator.sklearn_results.items():
        coef = sklearn_result["fitted"].coef_
        err = np.linalg.norm(image_flat - coef)
        print(f"  sklearn {name}: ||image - coef|| = {err:.4f}")

    for name, xorq_result in comparator.xorq_results.items():
        coef = xorq_result["fitted"].coef_
        err = np.linalg.norm(image_flat - coef)
        print(f"  xorq   {name}: ||image - coef|| = {err:.4f}")


def plot_results(comparator):
    _, image, _, n_angles = _generate_raw_data()

    sk_ridge = comparator.sklearn_results[RIDGE]["fitted"].coef_.reshape(IMAGE_SHAPE)
    sk_lasso = comparator.sklearn_results[LASSO]["fitted"].coef_.reshape(IMAGE_SHAPE)
    xo_ridge = comparator.xorq_results[RIDGE]["fitted"].coef_.reshape(IMAGE_SHAPE)
    xo_lasso = comparator.xorq_results[LASSO]["fitted"].coef_.reshape(IMAGE_SHAPE)

    image_flat = image.ravel()

    def _make_fig(rec_l2, rec_l1, label):
        l2_err = np.linalg.norm(image_flat - rec_l2.ravel())
        l1_err = np.linalg.norm(image_flat - rec_l1.ravel())
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image, cmap="gray", interpolation="nearest")
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(rec_l2, cmap="gray", interpolation="nearest")
        axes[1].set_title(f"Ridge (L2) err={l2_err:.2f}")
        axes[1].axis("off")
        axes[2].imshow(rec_l1, cmap="gray", interpolation="nearest")
        axes[2].set_title(f"Lasso (L1) err={l1_err:.2f}")
        axes[2].axis("off")
        fig.suptitle(f"{label} - Tomography: {n_angles} projections", fontsize=13)
        fig.tight_layout()
        return fig

    sk_fig = _make_fig(sk_ridge, sk_lasso, "sklearn")
    xo_fig = _make_fig(xo_ridge, xo_lasso, "xorq")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn")
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq")
    axes[1].axis("off")

    fig.suptitle(
        f"Tomography Reconstruction: {n_angles} projections, {N_PIXELS} pixels",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = (RIDGE, LASSO) = ("Ridge", "Lasso")
names_pipelines = (
    (RIDGE, SklearnPipeline([("ridge", Ridge(alpha=0.2))])),
    (LASSO, SklearnPipeline([("lasso", Lasso(alpha=0.001))])),
)
metrics_names_funcs = ()

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=TARGET_COL,
    pred=PRED_COL,
    metrics_names_funcs=metrics_names_funcs,
    load_data=load_data,
    split_data=split_data_nop,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)
# expose the exprs to invoke `xorq build plot_tomography_l1_reconstruction.py --expr $expr_name`
(xorq_ridge_preds, xorq_lasso_preds) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    comparator.result_comparison
    save_fig("imgs/tomography_l1_reconstruction.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
