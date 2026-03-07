"""Quantile Regression
===================

sklearn: Generate heteroscedastic Normal and asymmetric Pareto distributed targets
with linear mean. Fit QuantileRegressor at quantiles 0.05, 0.5, 0.95 eagerly,
identify outliers beyond the 90% interval, and compare with LinearRegression
via MAE and MSE metrics.

xorq: Same models wrapped in Pipeline.from_instance. Data is an ibis expression,
fit/predict deferred, metrics via deferred_sklearn_metric, outlier detection
via deferred boolean operations.

Both produce identical quantile predictions and metrics.

Dataset: Synthetic linear data with heteroscedastic Normal or asymmetric Pareto noise

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/linear_model/plot_quantile_regression.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline as SklearnPipeline

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    split_data_nop,
)
from xorq_gallery.utils import fig_to_image, save_fig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_SAMPLES = 100
FEATURE_COLS = ("feature_0",)
TARGET_COL_NORMAL = "y_normal"
TARGET_COL_PARETO = "y_pareto"
PRED_COL = "pred"

METRICS_NAMES_FUNCS = (
    ("mae", mean_absolute_error),
    ("mse", mean_squared_error),
)

PIPELINE_NAMES = ("pred_q05", "pred_q50", "pred_q95", "pred_lr", "pred_qr")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _generate_data():
    """Generate synthetic linear data with heteroscedastic/asymmetric noise."""
    rng = np.random.RandomState(RANDOM_STATE)
    x = np.linspace(start=0, stop=10, num=N_SAMPLES)
    y_true_mean = 10 + 0.5 * x

    y_normal = y_true_mean + rng.normal(loc=0, scale=0.5 + 0.5 * x, size=x.shape[0])

    a = 5
    y_pareto = y_true_mean + 10 * (rng.pareto(a, size=x.shape[0]) - 1 / (a - 1))

    return pd.DataFrame(
        {
            FEATURE_COLS[0]: x,
            TARGET_COL_NORMAL: y_normal,
            TARGET_COL_PARETO: y_pareto,
            "x": x,
            "y_true_mean": y_true_mean,
        }
    )


_DF = _generate_data()


def load_data_normal():
    return _DF[[FEATURE_COLS[0], TARGET_COL_NORMAL, "x", "y_true_mean"]].rename(
        columns={TARGET_COL_NORMAL: "y"}
    )


def load_data_pareto():
    return _DF[[FEATURE_COLS[0], TARGET_COL_PARETO, "x", "y_true_mean"]].rename(
        columns={TARGET_COL_PARETO: "y"}
    )


TARGET_COL = "y"

_LOAD_DATA_FNS = {
    "normal": load_data_normal,
    "pareto": load_data_pareto,
}


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

names_pipelines = (
    ("pred_q05", SklearnPipeline([("qr", QuantileRegressor(quantile=0.05, alpha=0))])),
    ("pred_q50", SklearnPipeline([("qr", QuantileRegressor(quantile=0.50, alpha=0))])),
    ("pred_q95", SklearnPipeline([("qr", QuantileRegressor(quantile=0.95, alpha=0))])),
    ("pred_lr", SklearnPipeline([("lr", LinearRegression())])),
    ("pred_qr", SklearnPipeline([("qr", QuantileRegressor(quantile=0.50, alpha=0))])),
)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _build_quantile_subplot(ax, df, preds_q05, preds_q50, preds_q95, title):
    """Plot quantile regression lines, data, and out-of-bounds markers."""
    x = df["x"].values if "x" in df.columns else df[FEATURE_COLS[0]].values
    y = df[TARGET_COL].values
    y_true_mean = df["y_true_mean"].values

    out_bounds = (preds_q05 >= y) | (preds_q95 <= y)

    ax.plot(x, y_true_mean, color="black", linestyle="dashed", label="True mean")
    ax.plot(x, preds_q05, label="Quantile: 0.05")
    ax.plot(x, preds_q50, label="Quantile: 0.5")
    ax.plot(x, preds_q95, label="Quantile: 0.95")

    ax.scatter(
        x[out_bounds],
        y[out_bounds],
        color="black",
        marker="+",
        alpha=0.5,
        label="Outside interval",
    )
    ax.scatter(
        x[~out_bounds],
        y[~out_bounds],
        color="black",
        alpha=0.5,
        label="Inside interval",
    )

    ax.legend(fontsize=7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=10)


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def compare_results(comparator):
    print("\n=== Comparing Results ===")
    for name in ("pred_lr", "pred_qr"):
        sk_mae = comparator.sklearn_results[name]["metrics"]["mae"]
        sk_mse = comparator.sklearn_results[name]["metrics"]["mse"]
        xo_mae = comparator.xorq_results[name]["metrics"]["mae"]
        xo_mse = comparator.xorq_results[name]["metrics"]["mse"]
        label = "LinearRegression" if name == "pred_lr" else "QuantileRegressor"
        print(f"  {label:22s} MAE — sklearn: {sk_mae:.3f}, xorq: {xo_mae:.3f}")
        print(f"  {label:22s} MSE — sklearn: {sk_mse:.3f}, xorq: {xo_mse:.3f}")
        np.testing.assert_allclose(sk_mae, xo_mae, rtol=1e-2)
        np.testing.assert_allclose(sk_mse, xo_mse, rtol=1e-2)


def plot_results(comparator):
    """Build a 2-row figure: sklearn (top) and xorq (bottom)."""
    df = comparator.df

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # sklearn row
    sk_q05 = comparator.sklearn_results["pred_q05"]["preds"]
    sk_q50 = comparator.sklearn_results["pred_q50"]["preds"]
    sk_q95 = comparator.sklearn_results["pred_q95"]["preds"]
    _build_quantile_subplot(axes[0], df, sk_q05, sk_q50, sk_q95, "sklearn")

    # xorq row
    xo_q05 = comparator.xorq_results["pred_q05"]["preds"][PRED_COL].values
    xo_q50 = comparator.xorq_results["pred_q50"]["preds"][PRED_COL].values
    xo_q95 = comparator.xorq_results["pred_q95"]["preds"][PRED_COL].values
    _build_quantile_subplot(axes[1], df, xo_q05, xo_q50, xo_q95, "xorq")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup — one comparator per dataset type
# ---------------------------------------------------------------------------

dataset_names = ("normal", "pareto")

comparators = {
    ds_name: SklearnXorqComparator(
        names_pipelines=names_pipelines,
        features=FEATURE_COLS,
        target=TARGET_COL,
        pred=PRED_COL,
        metrics_names_funcs=METRICS_NAMES_FUNCS,
        load_data=_LOAD_DATA_FNS[ds_name],
        split_data=split_data_nop,
        compare_results_fn=compare_results,
        plot_results_fn=plot_results,
    )
    for ds_name in dataset_names
}

# expose the exprs to invoke `xorq build plot_quantile_regression.py --expr $expr_name`
(
    xorq_normal_q05_preds,
    xorq_normal_q50_preds,
    xorq_normal_q95_preds,
    xorq_normal_lr_preds,
    xorq_normal_qr_preds,
) = (
    comparators["normal"].deferred_xorq_results[name]["preds"]
    for name in PIPELINE_NAMES
)
(
    xorq_pareto_q05_preds,
    xorq_pareto_q50_preds,
    xorq_pareto_q95_preds,
    xorq_pareto_lr_preds,
    xorq_pareto_qr_preds,
) = (
    comparators["pareto"].deferred_xorq_results[name]["preds"]
    for name in PIPELINE_NAMES
)


# =========================================================================
# Main
# =========================================================================


def _build_composite_figure():
    """Compose per-dataset figures into a single composite figure."""
    row_figs = [comparators[ds_name].plot_results() for ds_name in dataset_names]

    fig, axes = plt.subplots(1, len(dataset_names), figsize=(16, 8))
    for col, (row_fig, ds_name) in enumerate(zip(row_figs, dataset_names)):
        axes[col].imshow(fig_to_image(row_fig))
        axes[col].axis("off")
        axes[col].set_title(ds_name.capitalize(), fontsize=13, fontweight="bold")
        plt.close(row_fig)

    fig.suptitle("Quantile Regression: sklearn vs xorq", fontsize=16, y=0.98)
    fig.tight_layout()
    return fig


def main():
    for ds_name in dataset_names:
        print(f"\n=== DATASET: {ds_name.upper()} ===")
        comparators[ds_name].result_comparison
    save_fig("imgs/plot_quantile_regression.png", _build_composite_figure())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
