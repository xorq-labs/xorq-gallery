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
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline as SklearnPipeline
from toolz import curry

from xorq_gallery.sklearn.sklearn_lib import SklearnXorqComparator
from xorq_gallery.utils import (
    fig_to_image,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_SAMPLES = 100
FEATURE_COLS = ("feature_0",)
ROW_IDX = "row_idx"

METRICS_NAMES_FUNCS = (
    ("mae", mean_absolute_error),
    ("mse", mean_squared_error),
)


# ---------------------------------------------------------------------------
# Data generation (shared)
# ---------------------------------------------------------------------------


def _generate_datasets():
    """Generate two synthetic datasets with heteroscedastic and asymmetric noise.

    Returns
    -------
    dict with keys "normal" and "pareto", each containing:
        - "X": features (n_samples x 1)
        - "y": targets
        - "x": 1D x values for plotting
        - "y_true_mean": true conditional mean
    """
    rng = np.random.RandomState(RANDOM_STATE)
    x = np.linspace(start=0, stop=10, num=N_SAMPLES)
    X = x[:, np.newaxis]
    y_true_mean = 10 + 0.5 * x

    # Heteroscedastic Normal noise
    y_normal = y_true_mean + rng.normal(loc=0, scale=0.5 + 0.5 * x, size=x.shape[0])

    # Asymmetric Pareto noise
    a = 5
    y_pareto = y_true_mean + 10 * (rng.pareto(a, size=x.shape[0]) - 1 / (a - 1))

    return {
        "normal": {"X": X, "y": y_normal, "x": x, "y_true_mean": y_true_mean},
        "pareto": {"X": X, "y": y_pareto, "x": x, "y_true_mean": y_true_mean},
    }


def _load_data():
    """Load data as pandas DataFrame with row_idx for temporal ordering."""
    datasets = _generate_datasets()

    # Build DataFrame with both targets
    df = pd.DataFrame(datasets["normal"]["X"], columns=[FEATURE_COLS[0]]).assign(**{
        "y_normal": datasets["normal"]["y"],
        "y_pareto": datasets["pareto"]["y"],
        "x": datasets["normal"]["x"],
        "y_true_mean": datasets["normal"]["y_true_mean"],
        ROW_IDX: range(len(datasets["normal"]["X"])),
    })

    return df


# ---------------------------------------------------------------------------
# Plotting helpers for deferred_matplotlib_plot
# ---------------------------------------------------------------------------


@curry
def _build_quantile_plot(df, dataset_type, y_col):
    """Build quantile regression plot for specified dataset type.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: x, y_col, y_true_mean, pred_q05, pred_q50, pred_q95,
        out_bounds
    dataset_type : str
        Either "normal" or "pareto" (used for title)
    y_col : str
        Name of the target column to plot

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        df["x"], df["y_true_mean"], color="black", linestyle="dashed", label="True mean"
    )
    ax.plot(df["x"], df["pred_q05"], label="Quantile: 0.05")
    ax.plot(df["x"], df["pred_q50"], label="Quantile: 0.5")
    ax.plot(df["x"], df["pred_q95"], label="Quantile: 0.95")

    # Separate inside/outside interval
    out_mask = df["out_bounds"].astype(bool)
    ax.scatter(
        df.loc[out_mask, "x"],
        df.loc[out_mask, y_col],
        color="black",
        marker="+",
        alpha=0.5,
        label="Outside interval",
    )
    ax.scatter(
        df.loc[~out_mask, "x"],
        df.loc[~out_mask, y_col],
        color="black",
        alpha=0.5,
        label="Inside interval",
    )

    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if dataset_type == "normal":
        ax.set_title("Quantiles of heteroscedastic Normal distributed target")
    else:
        ax.set_title("Quantiles of asymmetric Pareto distributed target")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Comparator factory
# ---------------------------------------------------------------------------


def _make_comparator(sklearn_pipeline, input_expr, dataset_type, pred):
    return SklearnXorqComparator(
        sklearn_pipeline=sklearn_pipeline,
        input_expr=input_expr,
        kwargs_tuple=(
            ("features", FEATURE_COLS),
            ("target", f"y_{dataset_type}"),
            ("pred", pred),
        ),
        metrics_names_funcs=METRICS_NAMES_FUNCS,
    )


# ---------------------------------------------------------------------------
# Module-level data and comparator instances
# ---------------------------------------------------------------------------

_DF = _load_data()
INPUT_EXPR = xo.connect().register(_DF, "quantile_data")

PIPELINES = {
    "pred_q05": SklearnPipeline([("qr", QuantileRegressor(quantile=0.05, alpha=0))]),
    "pred_q50": SklearnPipeline([("qr", QuantileRegressor(quantile=0.50, alpha=0))]),
    "pred_q95": SklearnPipeline([("qr", QuantileRegressor(quantile=0.95, alpha=0))]),
    "pred_lr":  SklearnPipeline([("lr", LinearRegression())]),
    "pred_qr":  SklearnPipeline([("qr", QuantileRegressor(quantile=0.50, alpha=0))]),
}

_COMPARATORS = {
    dataset_type: {
        pred: _make_comparator(pipeline, INPUT_EXPR, dataset_type, pred)
        for pred, pipeline in PIPELINES.items()
    }
    for dataset_type in ("normal", "pareto")
}


# =========================================================================
# SKLEARN WAY -- eager execution
# =========================================================================


def sklearn_way(dataset_type="normal"):
    """Eager sklearn: fit quantile regressors at 0.05, 0.5, 0.95.

    Parameters
    ----------
    dataset_type : str
        Either "normal" or "pareto"

    Returns
    -------
    dict
        Keys: "predictions" (dict of quantile -> array), "out_bounds" (bool array),
              "lr_metrics" (dict with mae, mse)
    """
    comparators = _COMPARATORS[dataset_type]

    y_pred_05 = comparators["pred_q05"].sklearn_prediction
    y_pred_50 = comparators["pred_q50"].sklearn_prediction
    y_pred_95 = comparators["pred_q95"].sklearn_prediction
    out_bounds = (y_pred_05 >= comparators["pred_q05"].y) | (y_pred_95 <= comparators["pred_q05"].y)

    print(f"  sklearn LinearRegression:    MAE={comparators['pred_lr'].sklearn_metrics['mae']:.3f}, MSE={comparators['pred_lr'].sklearn_metrics['mse']:.3f}")
    print(f"  sklearn QuantileRegressor:   MAE={comparators['pred_qr'].sklearn_metrics['mae']:.3f}, MSE={comparators['pred_qr'].sklearn_metrics['mse']:.3f}")

    return {
        "predictions": {0.05: y_pred_05, 0.5: y_pred_50, 0.95: y_pred_95},
        "out_bounds": out_bounds,
        "lr_metrics": comparators["pred_lr"].sklearn_metrics,
        "qr_metrics": comparators["pred_qr"].sklearn_metrics,
    }


# =========================================================================
# XORQ WAY -- deferred execution
# =========================================================================


def xorq_way(dataset_type="normal"):
    """Deferred xorq: fit quantile regressors at 0.05, 0.5, 0.95.

    Parameters
    ----------
    dataset_type : str
        Either "normal" or "pareto"

    Returns deferred expressions for predictions, metrics, and plot.
    Nothing is executed until ``.execute()``.

    Returns
    -------
    dict
        Keys: "predictions", "lr_metrics", "qr_metrics"
    """
    comparators = _COMPARATORS[dataset_type]

    return {
        "predictions": {
            0.05: comparators["pred_q05"].xorq_prediction,
            0.5:  comparators["pred_q50"].xorq_prediction,
            0.95: comparators["pred_q95"].xorq_prediction,
        },
        "lr_metrics": comparators["pred_lr"].xorq_metrics,
        "qr_metrics": comparators["pred_qr"].xorq_metrics,
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    # We'll run both normal and pareto, but for brevity, focus on normal in main output
    for dataset_type in ["normal", "pareto"]:
        print(f"\n=== DATASET: {dataset_type.upper()} ===")

        print("=== SKLEARN WAY ===")
        sk_results = sklearn_way(dataset_type)

        print("\n=== XORQ WAY ===")
        deferred = xorq_way(dataset_type)

        # Execute deferred metrics
        lr_metrics_df = deferred["lr_metrics"].execute()
        qr_metrics_df = deferred["qr_metrics"].execute()

        xo_mae_lr = lr_metrics_df["mae"].iloc[0]
        xo_mse_lr = lr_metrics_df["mse"].iloc[0]
        xo_mae_qr = qr_metrics_df["mae"].iloc[0]
        xo_mse_qr = qr_metrics_df["mse"].iloc[0]

        print(f"  xorq   LinearRegression:    MAE={xo_mae_lr:.3f}, MSE={xo_mse_lr:.3f}")
        print(f"  xorq   QuantileRegressor:   MAE={xo_mae_qr:.3f}, MSE={xo_mse_qr:.3f}")

        # ---- Assert numerical equivalence BEFORE plotting ----
        np.testing.assert_allclose(
            sk_results["lr_metrics"]["mae"], xo_mae_lr, rtol=1e-2
        )
        np.testing.assert_allclose(
            sk_results["lr_metrics"]["mse"], xo_mse_lr, rtol=1e-2
        )
        np.testing.assert_allclose(
            sk_results["qr_metrics"]["mae"], xo_mae_qr, rtol=1e-2
        )
        np.testing.assert_allclose(
            sk_results["qr_metrics"]["mse"], xo_mse_qr, rtol=1e-2
        )
        print("Assertions passed: sklearn and xorq metrics match.")

        # Execute deferred predictions and build combined dataframe for plotting
        xo_predictions = {
            quantile: pred_expr.execute()[f"pred_q{int(quantile * 100):02d}"].values
            for quantile, pred_expr in deferred["predictions"].items()
        }

        # Build xorq plot dataframe
        target_col = f"y_{dataset_type}"
        xo_plot_df = _DF.assign(**{
            "pred_q05": xo_predictions[0.05],
            "pred_q50": xo_predictions[0.5],
            "pred_q95": xo_predictions[0.95],
            # Compute out_bounds flag for xorq predictions
            "out_bounds": (~_DF[target_col].between(xo_predictions[0.05], xo_predictions[0.95], inclusive="neither")).astype(int),
        })

        # Build sklearn plot
        sk_plot_df = _DF.assign(**{
            "pred_q05": sk_results["predictions"][0.05],
            "pred_q50": sk_results["predictions"][0.5],
            "pred_q95": sk_results["predictions"][0.95],
            "out_bounds": sk_results["out_bounds"].astype(int),
        })

        sk_fig = _build_quantile_plot(sk_plot_df, dataset_type, target_col)
        xo_fig = _build_quantile_plot(xo_plot_df, dataset_type, target_col)

        # Composite: sklearn (left) | xorq (right)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].imshow(fig_to_image(sk_fig))
        axes[0].axis("off")
        axes[1].imshow(fig_to_image(xo_fig))
        axes[1].axis("off")

        fig.suptitle(
            f"Quantile Regression ({dataset_type}): sklearn vs xorq", fontsize=16
        )
        fig.tight_layout()
        out = f"imgs/quantile_regression_{dataset_type}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
