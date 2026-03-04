"""Compare the effect of different scalers on data with outliers
=============================================================

sklearn: Apply StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,
PowerTransformer (Yeo-Johnson and Box-Cox), QuantileTransformer (uniform and
normal), and Normalizer to California Housing features (MedInc, AveOccup).
Each scaler is fit and transformed eagerly on numpy arrays.

xorq: Same scalers wrapped in Pipeline.from_instance, deferred execution via
xorq. Each transformation is computed lazily as a deferred expression and
materialized only when executed.

Both produce identical scaled data for visualization.

Dataset: California Housing (sklearn.datasets)
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from matplotlib import cm
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    minmax_scale,
)
from xorq.api import SessionConfig
from xorq.config import options

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    make_deferred_xorq_fit_transform_result,
    make_sklearn_fit_transform_result,
    make_xorq_fit_transform_result,
    split_data_nop,
)
from xorq_gallery.utils import fig_to_image, save_fig


# Force single-threaded DataFusion to preserve scan order
options.backend = xo.connect(session_config=SessionConfig().with_target_partitions(1))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS = ("MedInc", "AveOccup")
TARGET_COL = "y_scaled"
PRED_COL = "pred"  # unused, required by comparator

FEATURE_LABELS = ("Median income in block", "Average house occupancy")

# plasma_r colormap for consistency with sklearn example
CMAP = getattr(cm, "plasma_r", cm.hot_r)

# QuantileTransformer uses random subsampling internally; row-ordering
# differences between ibis and pandas cause small deviations — use tolerances.
COMPARE_RTOL = 1e-7
COMPARE_ATOL = 1e-7  # tightened to test if single-partition fixes QuantileTransformer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Load California Housing and return flat DataFrame with two features + target."""
    dataset = fetch_california_housing()
    feature_names = dataset.feature_names
    features_idx = tuple(feature_names.index(f) for f in FEATURE_COLS)
    X = dataset.data[:, features_idx]
    y_scaled = minmax_scale(dataset.target)

    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df[TARGET_COL] = y_scaled
    df.attrs["y_full_range"] = (dataset.target.min(), dataset.target.max())
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _create_axes(figsize=(16, 6)):
    plt.figure(figsize=figsize)
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    ax_scatter = plt.axes([left, bottom, width, height])
    ax_histx = plt.axes([left, bottom_h, width, 0.1])
    ax_histy = plt.axes([left_h, bottom, 0.05, height])

    left = width + left + 0.2
    left_h = left + width + 0.02

    ax_scatter_zoom = plt.axes([left, bottom, width, height])
    ax_histx_zoom = plt.axes([left, bottom_h, width, 0.1])
    ax_histy_zoom = plt.axes([left_h, bottom, 0.05, height])

    left, width = width + left + 0.13, 0.01
    ax_colorbar = plt.axes([left, bottom, width, height])

    return (
        (ax_scatter, ax_histy, ax_histx),
        (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
        ax_colorbar,
    )


def _plot_distribution(axes, X, y, hist_nbins=50, title="", x0_label="", x1_label=""):
    ax, hist_X1, hist_X0 = axes
    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)
    colors = CMAP(y)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker="o", s=5, lw=0, c=colors)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(
        X[:, 1], bins=hist_nbins, orientation="horizontal", color="grey", ec="grey"
    )
    hist_X1.axis("off")
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(
        X[:, 0], bins=hist_nbins, orientation="vertical", color="grey", ec="grey"
    )
    hist_X0.axis("off")


def _make_plot(X, y, title, y_full_min, y_full_max):
    ax_zoom_out, ax_zoom_in, ax_colorbar = _create_axes()
    _plot_distribution(
        ax_zoom_out,
        X,
        y,
        hist_nbins=200,
        x0_label=FEATURE_LABELS[0],
        x1_label=FEATURE_LABELS[1],
        title="Full data",
    )
    zoom_in_percentile_range = (0, 99)
    cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)
    non_outliers_mask = np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) & np.all(
        X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1
    )
    _plot_distribution(
        ax_zoom_in,
        X[non_outliers_mask],
        y[non_outliers_mask],
        hist_nbins=50,
        x0_label=FEATURE_LABELS[0],
        x1_label=FEATURE_LABELS[1],
        title="Zoom-in",
    )
    norm = mpl.colors.Normalize(y_full_min, y_full_max)
    mpl.colorbar.ColorbarBase(
        ax_colorbar,
        cmap=CMAP,
        norm=norm,
        orientation="vertical",
        label="Color mapping for values of y",
    )
    fig = plt.gcf()
    fig.suptitle(title)
    return fig


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def compare_results(comparator):
    df = comparator.df
    X_baseline = df[list(FEATURE_COLS)].values
    print("\n=== Comparing Results ===")

    # Unscaled baseline — no sklearn/xorq result, just report shape
    print(f"  Unscaled data: shape {X_baseline.shape}")

    sklearn_stats = []
    xorq_stats = []
    for name in comparator.sklearn_results:
        sk_X = comparator.sklearn_results[name]["transformed"]
        xo_X = comparator.xorq_results[name]["transformed"][list(FEATURE_COLS)].values

        for label, X in (("sklearn", sk_X), ("xorq", xo_X)):
            print(
                f"  {label} {name}: "
                f"mean=({X[:, 0].mean():.3f}, {X[:, 1].mean():.3f})  "
                f"std=({X[:, 0].std():.3f}, {X[:, 1].std():.3f})"
            )

        sklearn_stats.append(
            {
                "scaler": name,
                "mean_feature0": sk_X[:, 0].mean(),
                "std_feature0": sk_X[:, 0].std(),
                "mean_feature1": sk_X[:, 1].mean(),
                "std_feature1": sk_X[:, 1].std(),
            }
        )
        xorq_stats.append(
            {
                "scaler": name,
                "mean_feature0": xo_X[:, 0].mean(),
                "std_feature0": xo_X[:, 0].std(),
                "mean_feature1": xo_X[:, 1].mean(),
                "std_feature1": xo_X[:, 1].std(),
            }
        )

    # QuantileTransformer uses random subsampling; use tolerances (not exact equality)
    pd.testing.assert_frame_equal(
        pd.DataFrame(sklearn_stats).set_index("scaler"),
        pd.DataFrame(xorq_stats).set_index("scaler"),
        rtol=COMPARE_RTOL,
        atol=COMPARE_ATOL,
        check_dtype=False,
    )
    print(
        f"  All {len(sklearn_stats)} scalers: sklearn vs xorq stats match (rtol={COMPARE_RTOL}, atol={COMPARE_ATOL})"
    )


# Selected scalers for the composite plot (subset keeps output manageable)
PLOT_NAMES = (
    "standard",
    "minmax",
    "robust",
    "power_yj",
    "quantile_uniform",
)


def plot_results(comparator):
    df = comparator.df
    y = df[TARGET_COL].values
    y_full_min, y_full_max = df.attrs["y_full_range"]
    X_baseline = df[list(FEATURE_COLS)].values

    all_names = ("unscaled",) + PLOT_NAMES
    n = len(all_names)
    fig, axes = plt.subplots(n, 2, figsize=(32, 6 * n))

    for row, name in enumerate(all_names):
        if name == "unscaled":
            sk_X = xo_X = X_baseline
            xo_y = y
            title = "Unscaled data"
        else:
            sk_X = comparator.sklearn_results[name]["transformed"]
            xo_result = comparator.xorq_results[name]["transformed"]
            xo_X = xo_result[list(FEATURE_COLS)].values
            xo_y = xo_result[TARGET_COL].values
            title = name

        sk_fig = _make_plot(sk_X, y, f"sklearn: {title}", y_full_min, y_full_max)
        xo_fig = _make_plot(xo_X, xo_y, f"xorq: {title}", y_full_min, y_full_max)

        axes[row, 0].imshow(fig_to_image(sk_fig))
        axes[row, 0].axis("off")
        axes[row, 1].imshow(fig_to_image(xo_fig))
        axes[row, 1].axis("off")

    axes[0, 0].set_title("sklearn", fontsize=14, fontweight="bold", pad=10)
    axes[0, 1].set_title("xorq", fontsize=14, fontweight="bold", pad=10)
    fig.suptitle("Compare scalers on California Housing: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = (
    STANDARD,
    MINMAX,
    MAXABS,
    ROBUST,
    POWER_YJ,
    POWER_BC,
    QUANTILE_UNIFORM,
    QUANTILE_NORMAL,
    NORMALIZER,
) = (
    "standard",
    "minmax",
    "maxabs",
    "robust",
    "power_yj",
    "power_bc",
    "quantile_uniform",
    "quantile_normal",
    "normalizer",
)

names_pipelines = (
    (STANDARD, SklearnPipeline([("scaler", StandardScaler())])),
    (MINMAX, SklearnPipeline([("scaler", MinMaxScaler())])),
    (MAXABS, SklearnPipeline([("scaler", MaxAbsScaler())])),
    (ROBUST, SklearnPipeline([("scaler", RobustScaler(quantile_range=(25, 75)))])),
    (POWER_YJ, SklearnPipeline([("scaler", PowerTransformer(method="yeo-johnson"))])),
    (POWER_BC, SklearnPipeline([("scaler", PowerTransformer(method="box-cox"))])),
    (
        QUANTILE_UNIFORM,
        SklearnPipeline(
            [
                (
                    "scaler",
                    QuantileTransformer(output_distribution="uniform", random_state=42),
                )
            ]
        ),
    ),
    (
        QUANTILE_NORMAL,
        SklearnPipeline(
            [
                (
                    "scaler",
                    QuantileTransformer(output_distribution="normal", random_state=42),
                )
            ]
        ),
    ),
    (NORMALIZER, SklearnPipeline([("scaler", Normalizer())])),
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
    make_sklearn_result=make_sklearn_fit_transform_result,
    make_deferred_xorq_result=make_deferred_xorq_fit_transform_result,
    make_xorq_result=make_xorq_fit_transform_result,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)
# expose the exprs to invoke `xorq build plot_all_scaling.py --expr $expr_name`
(
    xorq_standard_transformed,
    xorq_minmax_transformed,
    xorq_maxabs_transformed,
    xorq_robust_transformed,
    xorq_power_yj_transformed,
    xorq_power_bc_transformed,
    xorq_quantile_uniform_transformed,
    xorq_quantile_normal_transformed,
    xorq_normalizer_transformed,
) = (comparator.deferred_xorq_results[name]["transformed"] for name in methods)


def main():
    comparator.result_comparison
    save_fig("imgs/all_scaling.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
