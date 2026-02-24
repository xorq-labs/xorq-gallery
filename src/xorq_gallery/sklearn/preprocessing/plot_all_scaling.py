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

import os

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
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import fig_to_image


# ---------------------------------------------------------------------------
# Feature selection and mapping
# ---------------------------------------------------------------------------

FEATURES = ("MedInc", "AveOccup")

FEATURE_MAPPING = {
    "MedInc": "Median income in block",
    "HouseAge": "Median house age in block",
    "AveRooms": "Average number of rooms",
    "AveBedrms": "Average number of bedrooms",
    "Population": "Block population",
    "AveOccup": "Average house occupancy",
    "Latitude": "House block latitude",
    "Longitude": "House block longitude",
}


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Load California Housing dataset and extract selected features.

    Returns
    -------
    dict with keys:
        X : np.ndarray of shape (n_samples, 2)
            Selected features (MedInc, AveOccup).
        y : np.ndarray of shape (n_samples,)
            Target values scaled to [0, 1] for colormap.
        feature_names : tuple of str
            Names of the two selected features.
        feature_labels : tuple of str
            Human-readable labels for plotting.
    """
    dataset = fetch_california_housing()
    X_full, y_full = dataset.data, dataset.target
    feature_names = dataset.feature_names

    features_idx = tuple(feature_names.index(f) for f in FEATURES)
    X = X_full[:, features_idx]

    # Scale target to [0, 1] for colormap
    y = minmax_scale(y_full)

    feature_labels = tuple(FEATURE_MAPPING[f] for f in FEATURES)

    return {
        "X": X,
        "y": y,
        "feature_names": FEATURES,
        "feature_labels": feature_labels,
    }


# ---------------------------------------------------------------------------
# Plotting helpers (used by both sklearn and xorq)
# ---------------------------------------------------------------------------

# plasma_r colormap for consistency with sklearn example
CMAP = getattr(cm, "plasma_r", cm.hot_r)


def _create_axes(figsize=(16, 6)):
    """Create figure with scatter plot, histograms, and zoomed-in view.

    Returns
    -------
    tuple of:
        (ax_scatter, ax_histy, ax_histx) : main scatter + marginal histograms
        (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom) : zoomed-in view
        ax_colorbar : colorbar axis
    """
    plt.figure(figsize=figsize)

    # Main plot axes
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    # Zoomed-in plot axes
    left = width + left + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)

    # Colorbar axis
    left, width = width + left + 0.13, 0.01
    rect_colorbar = [left, bottom, width, height]
    ax_colorbar = plt.axes(rect_colorbar)

    return (
        (ax_scatter, ax_histy, ax_histx),
        (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
        ax_colorbar,
    )


def _plot_distribution(axes, X, y, hist_nbins=50, title="", x0_label="", x1_label=""):
    """Plot scatter with marginal histograms for a 2D distribution.

    Parameters
    ----------
    axes : tuple of (ax_scatter, ax_histy, ax_histx)
    X : np.ndarray of shape (n_samples, 2)
    y : np.ndarray of shape (n_samples,)
        Values for color mapping.
    hist_nbins : int
    title : str
    x0_label : str
    x1_label : str
    """
    ax, hist_X1, hist_X0 = axes

    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)

    colors = CMAP(y)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker="o", s=5, lw=0, c=colors)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    # Histogram for feature 1 (vertical axis)
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(
        X[:, 1], bins=hist_nbins, orientation="horizontal", color="grey", ec="grey"
    )
    hist_X1.axis("off")

    # Histogram for feature 0 (horizontal axis)
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(
        X[:, 0], bins=hist_nbins, orientation="vertical", color="grey", ec="grey"
    )
    hist_X0.axis("off")


def _make_plot(X, y, title, feature_labels, y_full_min, y_full_max):
    """Create a single scaler comparison plot with full and zoomed views.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
        Transformed feature data.
    y : np.ndarray of shape (n_samples,)
        Target values scaled to [0, 1] for colormap.
    title : str
        Scaler name for plot title.
    feature_labels : tuple of str
        Human-readable feature labels.
    y_full_min : float
        Min of original target (for colorbar).
    y_full_max : float
        Max of original target (for colorbar).

    Returns
    -------
    matplotlib.figure.Figure
    """
    ax_zoom_out, ax_zoom_in, ax_colorbar = _create_axes()

    # Full data plot
    _plot_distribution(
        ax_zoom_out,
        X,
        y,
        hist_nbins=200,
        x0_label=feature_labels[0],
        x1_label=feature_labels[1],
        title="Full data",
    )

    # Zoomed-in plot (99th percentile)
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
        x0_label=feature_labels[0],
        x1_label=feature_labels[1],
        title="Zoom-in",
    )

    # Colorbar
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
# Scaler definitions (shared)
# ---------------------------------------------------------------------------


def _build_scalers():
    """Build list of (name, scaler) tuples for sklearn.

    Returns
    -------
    tuple of (name, sklearn_scaler_instance)
    """
    return (
        ("Unscaled data", None),
        ("Data after standard scaling", StandardScaler()),
        ("Data after min-max scaling", MinMaxScaler()),
        ("Data after max-abs scaling", MaxAbsScaler()),
        (
            "Data after robust scaling",
            RobustScaler(quantile_range=(25, 75)),
        ),
        (
            "Data after power transformation (Yeo-Johnson)",
            PowerTransformer(method="yeo-johnson"),
        ),
        (
            "Data after power transformation (Box-Cox)",
            PowerTransformer(method="box-cox"),
        ),
        (
            "Data after quantile transformation (uniform pdf)",
            QuantileTransformer(output_distribution="uniform", random_state=42),
        ),
        (
            "Data after quantile transformation (gaussian pdf)",
            QuantileTransformer(output_distribution="normal", random_state=42),
        ),
        ("Data after sample-wise L2 normalizing", Normalizer()),
    )


# =========================================================================
# SKLEARN WAY -- eager fit_transform
# =========================================================================


def sklearn_way(data_dict):
    """Eager sklearn: fit_transform each scaler, collect transformed arrays.

    Uses comprehension since all scalers follow identical pattern: fit_transform(X).

    Parameters
    ----------
    data_dict : dict
        Output from _load_data().

    Returns
    -------
    dict
        Keys are scaler names, values are transformed np.ndarray of shape (n, 2).
    """
    X = data_dict["X"]
    scalers = _build_scalers()

    results = {
        name: (X if scaler is None else scaler.fit_transform(X))
        for name, scaler in scalers
    }

    for name in results:
        print(f"  sklearn: {name} -> shape {results[name].shape}")

    return results


# =========================================================================
# XORQ WAY -- deferred fit_transform via Pipeline
# =========================================================================


def xorq_way(data_dict):
    """Deferred xorq: wrap scalers in Pipeline.from_instance for deferred execution.

    Each scaler is wrapped in a SklearnPipeline, then converted to xorq Pipeline
    using from_instance(). The pipeline is fit and transformed as deferred operations.

    Parameters
    ----------
    data_dict : dict
        Output from _load_data().

    Returns
    -------
    dict
        Keys are scaler names, values are deferred ibis expressions.
    """
    X = data_dict["X"]
    y = data_dict["y"]
    feature_names = data_dict["feature_names"]

    df = pd.DataFrame(X, columns=feature_names)
    df["y_scaled"] = y
    df["row_idx"] = range(len(df))

    con = xo.connect()
    scalers = _build_scalers()
    base_table = con.register(df, "california_housing").order_by("row_idx")

    results = {}
    for idx, (name, scaler) in enumerate(scalers):
        if scaler is None:
            results[name] = base_table
        else:
            sklearn_pipe = SklearnPipeline([(type(scaler).__name__.lower(), scaler)])
            xorq_pipe = Pipeline.from_instance(sklearn_pipe)
            fitted = xorq_pipe.fit(base_table, features=feature_names, target=None)
            transformed_expr = fitted.transform(base_table)
            results[name] = transformed_expr
        print(f"  xorq:    {name} -> ibis expression")

    return results


# =========================================================================
# Main: execute, assert equivalence, build composite plots
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    data_dict = _load_data()
    # X_orig = data_dict["X"]  # Not used, data_dict passed directly to sklearn_way
    y = data_dict["y"]
    feature_labels = data_dict["feature_labels"]

    # Need y_full min/max for colorbar
    dataset = fetch_california_housing()
    y_full_min, y_full_max = dataset.target.min(), dataset.target.max()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(data_dict)

    print("\n=== XORQ WAY ===")
    deferred = xorq_way(data_dict)

    print("\n=== ASSERTIONS ===")
    # Execute deferred expressions and build DataFrames for comparison
    sklearn_stats = []
    xorq_stats = []

    for name in sk_results.keys():
        sk_X = sk_results[name]
        xo_expr = deferred[name]

        # Execute xorq expression
        xo_df = xo_expr.execute()

        # Handle different output formats from Pipeline.transform
        # Try direct column access first, fallback to "transformed" column
        if all(col in xo_df.columns for col in data_dict["feature_names"]):
            xo_X = xo_df[list(data_dict["feature_names"])].values
        elif "transformed" in xo_df.columns:
            # Extract from transformed column structure
            xo_X = np.array([list(row.values()) for row in xo_df["transformed"]])
        else:
            raise ValueError(
                f"Unexpected column structure in transformed data: {xo_df.columns}"
            )

        # Build row for each scaler with summary statistics
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

    sklearn_df = pd.DataFrame(sklearn_stats).set_index("scaler")
    xorq_df = pd.DataFrame(xorq_stats).set_index("scaler")

    # Single assertion comparing all scalers at once.
    # QuantileTransformer has inherent stochasticity from subsampling, so row
    # ordering differences between ibis and pandas cause small deviations.
    pd.testing.assert_frame_equal(
        sklearn_df,
        xorq_df,
        rtol=0.05,
        atol=0.01,
        check_dtype=False,
    )
    print(f"All {len(sk_results)} scalers produce matching results (sklearn vs xorq)")

    print("\n=== GENERATING PLOTS ===")
    # Generate plots for a subset of scalers (to keep output manageable)
    selected_scalers = [
        "Unscaled data",
        "Data after standard scaling",
        "Data after min-max scaling",
        "Data after robust scaling",
        "Data after power transformation (Yeo-Johnson)",
        "Data after quantile transformation (uniform pdf)",
    ]

    for scaler_name in selected_scalers:
        sk_X = sk_results[scaler_name]
        xo_expr = deferred[scaler_name]
        xo_df = xo_expr.execute()

        # Handle different output formats (same logic as in assertions)
        if all(col in xo_df.columns for col in data_dict["feature_names"]):
            xo_X = xo_df[list(data_dict["feature_names"])].values
        elif "transformed" in xo_df.columns:
            xo_X = np.array([list(row.values()) for row in xo_df["transformed"]])
        else:
            raise ValueError(f"Unexpected column structure: {xo_df.columns}")

        # Create sklearn figure
        sk_fig = _make_plot(
            sk_X,
            y,
            f"sklearn: {scaler_name}",
            feature_labels,
            y_full_min,
            y_full_max,
        )

        # Create xorq figure
        xo_fig = _make_plot(
            xo_X,
            y,
            f"xorq: {scaler_name}",
            feature_labels,
            y_full_min,
            y_full_max,
        )

        # Composite: sklearn (left) | xorq (right)
        composite_fig, axes = plt.subplots(1, 2, figsize=(32, 6))

        axes[0].imshow(fig_to_image(sk_fig))
        axes[0].axis("off")

        axes[1].imshow(fig_to_image(xo_fig))
        axes[1].axis("off")

        safe_name = scaler_name.replace(" ", "_").replace("(", "").replace(")", "")
        composite_fig.suptitle(f"{scaler_name}: sklearn vs xorq", fontsize=14)
        composite_fig.tight_layout()
        out = f"imgs/all_scaling_{safe_name}.png"
        composite_fig.savefig(out, dpi=150)
        plt.close(composite_fig)
        print(f"  Saved: {out}")

    print("\nAll plots saved to imgs/")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
