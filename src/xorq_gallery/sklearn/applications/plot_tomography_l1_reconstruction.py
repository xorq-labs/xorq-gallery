"""Compressive sensing: tomography reconstruction with L1 prior (Lasso)
=======================================================================

sklearn: Build a sparse projection operator simulating a CT scanner,
generate noisy projections, solve the inverse problem with Ridge (L2) and
Lasso (L1) directly on numpy arrays. Compare reconstructions visually.

xorq: Register the projection matrix as an ibis table, fit Ridge and
Lasso via Pipeline.from_instance. Same results, deferred execution.

Dataset: Synthetic binary image (128x128)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from scipy import ndimage, sparse
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline as SklearnPipeline
from toolz import curry
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import deferred_matplotlib_plot, fig_to_image, load_plot_bytes


# ---------------------------------------------------------------------------
# Shared: synthetic data generation
# ---------------------------------------------------------------------------

img_size = 128
n_pixels = img_size * img_size
np.random.seed(0)


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


def _generate_data():
    n_angles = img_size // 7
    proj_operator = build_projection_operator(img_size, n_angles)
    image = generate_synthetic_data(img_size)
    proj = proj_operator @ image.ravel()[:, np.newaxis]
    proj += 0.15 * np.random.randn(*proj.shape)
    proj_flat = proj.ravel()
    return proj_operator, image, proj_flat, n_angles


# ---------------------------------------------------------------------------
# Plotting helpers (used by deferred_matplotlib_plot)
# ---------------------------------------------------------------------------


@curry
def _build_reconstruction_figure(_df, image, rec_l2, rec_l1, l2_err, l1_err, n_angles):
    """A UDAF-compatible plotting function for reconstruction results."""
    (image, rec_l2, rec_l1) = map(np.array, (image, rec_l2, rec_l1))
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

    fig.suptitle(f"xorq - Tomography: {n_angles} projections", fontsize=13)
    fig.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY
# =========================================================================


def sklearn_way(proj_operator, image, proj_flat):
    """Eager sklearn: fit Ridge and Lasso directly on sparse matrix."""
    ridge = Ridge(alpha=0.2)
    ridge.fit(proj_operator, proj_flat)
    rec_l2 = ridge.coef_.reshape(img_size, img_size)

    lasso = Lasso(alpha=0.001)
    lasso.fit(proj_operator, proj_flat)
    rec_l1 = lasso.coef_.reshape(img_size, img_size)

    l2_err = np.linalg.norm(image.ravel() - rec_l2.ravel())
    l1_err = np.linalg.norm(image.ravel() - rec_l1.ravel())
    print(f"  sklearn Ridge L2 error: {l2_err:.2f}")
    print(f"  sklearn Lasso L1 error: {l1_err:.2f}")

    return {"l2": rec_l2, "l1": rec_l1}


# =========================================================================
# XORQ WAY
# =========================================================================


def xorq_way(proj_operator, image, proj_flat, n_angles):
    """Deferred xorq: register matrix as ibis table, fit via Pipeline.

    Returns a deferred plot expression (PNG bytes via UDAF).  The fit
    goes through ibis data, then model coefficients are captured in the
    closure for the deferred visualization.
    """
    con = xo.connect()

    # Dense matrix as ibis table (small enough for this demo)
    proj_dense = proj_operator.toarray()
    pixel_cols = [f"px_{i}" for i in range(n_pixels)]
    tbl = pd.DataFrame(proj_dense, columns=pixel_cols)
    tbl["projection"] = proj_flat
    data = con.register(tbl, "tomography")

    features = tuple(pixel_cols)
    target_col = "projection"

    # Ridge (L2)
    xorq_ridge = Pipeline.from_instance(SklearnPipeline([("ridge", Ridge(alpha=0.2))]))
    fitted_ridge = xorq_ridge.fit(data, features=features, target=target_col)
    rec_l2 = fitted_ridge.predict_step.model.coef_.reshape(img_size, img_size)

    # Lasso (L1)
    xorq_lasso = Pipeline.from_instance(
        SklearnPipeline([("lasso", Lasso(alpha=0.001))])
    )
    fitted_lasso = xorq_lasso.fit(data, features=features, target=target_col)
    rec_l1 = fitted_lasso.predict_step.model.coef_.reshape(img_size, img_size)

    l2_err = np.linalg.norm(image.ravel() - rec_l2.ravel())
    l1_err = np.linalg.norm(image.ravel() - rec_l1.ravel())
    print(f"  xorq   Ridge L2 error: {l2_err:.2f}")
    print(f"  xorq   Lasso L1 error: {l1_err:.2f}")

    plot_fn = _build_reconstruction_figure(
        image=tuple(tuple(el) for el in image),
        rec_l2=tuple(tuple(el) for el in rec_l2),
        rec_l1=tuple(tuple(el) for el in rec_l1),
        l2_err=l2_err,
        l1_err=l1_err,
        n_angles=n_angles,
    )
    return deferred_matplotlib_plot(data, plot_fn)


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    proj_operator, image, proj_flat, n_angles = _generate_data()

    print("=== SKLEARN WAY ===")
    sk = sklearn_way(proj_operator, image, proj_flat)

    print("\n=== XORQ WAY ===")
    plot_expr = xorq_way(proj_operator, image, proj_flat, n_angles)

    # Execute deferred plot
    xo_png = plot_expr.execute()

    # Build sklearn figure natively
    l2_err = np.linalg.norm(image.ravel() - sk["l2"].ravel())
    l1_err = np.linalg.norm(image.ravel() - sk["l1"].ravel())

    sk_fig, sk_axes = plt.subplots(1, 3, figsize=(12, 4))
    sk_axes[0].imshow(image, cmap="gray", interpolation="nearest")
    sk_axes[0].set_title("Original")
    sk_axes[0].axis("off")
    sk_axes[1].imshow(sk["l2"], cmap="gray", interpolation="nearest")
    sk_axes[1].set_title(f"Ridge (L2) err={l2_err:.2f}")
    sk_axes[1].axis("off")
    sk_axes[2].imshow(sk["l1"], cmap="gray", interpolation="nearest")
    sk_axes[2].set_title(f"Lasso (L1) err={l1_err:.2f}")
    sk_axes[2].axis("off")
    sk_fig.suptitle(f"sklearn - Tomography: {n_angles} projections", fontsize=13)
    sk_fig.tight_layout()

    # Composite: sklearn (top) | xorq deferred (bottom)
    xo_img = load_plot_bytes(xo_png)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn")
    axes[0].axis("off")
    axes[1].imshow(xo_img)
    axes[1].set_title("xorq")
    axes[1].axis("off")

    fig.suptitle(
        f"Tomography Reconstruction: {n_angles} projections, {n_pixels} pixels",
        fontsize=13,
    )
    fig.tight_layout()
    out = "imgs/tomography_l1_reconstruction.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
