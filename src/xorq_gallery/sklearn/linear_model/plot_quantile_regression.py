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
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline as SklearnPipeline
from toolz import curry
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

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
PRED_COL = "pred"


# ---------------------------------------------------------------------------
# Data generation
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
    """Load data as pandas DataFrame with both target columns."""
    datasets = _generate_datasets()

    df = pd.DataFrame(datasets["normal"]["X"], columns=[FEATURE_COLS[0]])
    df["y_normal"] = datasets["normal"]["y"]
    df["y_pareto"] = datasets["pareto"]["y"]
    df["x"] = datasets["normal"]["x"]
    df["y_true_mean"] = datasets["normal"]["y_true_mean"]
    df[ROW_IDX] = range(len(df))

    return df


# ---------------------------------------------------------------------------
# Plot helper
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
# Expr builder: defined here — reader sees what gets built
# ---------------------------------------------------------------------------


def build_exprs_with_metrics(target_col):
    """Return a build_exprs_fn that does fit + predict + mae/mse metrics."""

    def _build(sklearn_pipeline, train_expr, test_expr, features, target, pred_name):
        pipeline = Pipeline.from_instance(sklearn_pipeline)
        fitted = pipeline.fit(train_expr, features=features, target=target)
        preds = fitted.predict(test_expr, name=pred_name)
        make_metric = deferred_sklearn_metric(target=target_col, pred=pred_name)
        return {
            "fitted_pipeline": fitted,
            "preds": preds,
            "metrics": preds.agg(
                mae=make_metric(metric=mean_absolute_error),
                mse=make_metric(metric=mean_squared_error),
            ),
        }

    return _build


# ---------------------------------------------------------------------------
# Compute: defined here — reader sees every sklearn and xorq call
# ---------------------------------------------------------------------------


def compute_sklearn(comparator):
    """Eager sklearn: fit quantile + linear regressors, compute metrics."""
    X = comparator.X
    y = comparator.y

    results = {}
    for name in comparator.names:
        pipe = comparator.sklearn_pipeline(name)
        fitted = pipe.fit(X, y)
        preds = fitted.predict(X)
        results[name] = {"predictions": preds}

    # Compute out_bounds flag from quantile predictions
    out_bounds = (results["QR_05"]["predictions"] >= y) | (
        results["QR_95"]["predictions"] <= y
    )
    results["out_bounds"] = out_bounds

    # Compute comparison metrics: LR vs QR_median
    y_pred_lr = results["LR"]["predictions"]
    y_pred_qr = results["QR_median"]["predictions"]

    mae_lr = mean_absolute_error(y, y_pred_lr)
    mse_lr = mean_squared_error(y, y_pred_lr)
    mae_qr = mean_absolute_error(y, y_pred_qr)
    mse_qr = mean_squared_error(y, y_pred_qr)

    results["lr_metrics"] = {"mae": mae_lr, "mse": mse_lr}
    results["qr_metrics"] = {"mae": mae_qr, "mse": mse_qr}

    print(f"  sklearn LinearRegression:    MAE={mae_lr:.3f}, MSE={mse_lr:.3f}")
    print(f"  sklearn QuantileRegressor:   MAE={mae_qr:.3f}, MSE={mse_qr:.3f}")

    return results


def compute_xorq(comparator):
    """Deferred xorq: execute deferred predictions and metrics."""
    exprs = comparator.deferred_exprs
    results = {}

    for name in comparator.names:
        e = exprs[name]
        pred_df = e["preds"].execute()
        pred_col = f"{comparator.pred_col}_{name}"
        results[name] = {"predictions": pred_df[pred_col].values}

    # Compute out_bounds flag from quantile predictions
    y = comparator.y
    out_bounds = (results["QR_05"]["predictions"] >= y) | (
        results["QR_95"]["predictions"] <= y
    )
    results["out_bounds"] = out_bounds

    # Execute deferred metrics for LR and QR_median
    lr_metrics_df = exprs["LR"]["metrics"].execute()
    qr_metrics_df = exprs["QR_median"]["metrics"].execute()

    xo_mae_lr = lr_metrics_df["mae"].iloc[0]
    xo_mse_lr = lr_metrics_df["mse"].iloc[0]
    xo_mae_qr = qr_metrics_df["mae"].iloc[0]
    xo_mse_qr = qr_metrics_df["mse"].iloc[0]

    results["lr_metrics"] = {"mae": xo_mae_lr, "mse": xo_mse_lr}
    results["qr_metrics"] = {"mae": xo_mae_qr, "mse": xo_mse_qr}

    print(f"  xorq   LinearRegression:    MAE={xo_mae_lr:.3f}, MSE={xo_mse_lr:.3f}")
    print(f"  xorq   QuantileRegressor:   MAE={xo_mae_qr:.3f}, MSE={xo_mse_qr:.3f}")

    return results


# ---------------------------------------------------------------------------
# Assertions: defined here — reader sees what gets compared
# ---------------------------------------------------------------------------


def build_assertions(sk, xo, comparator):
    """Build assertion pairs for LR vs QR metrics."""
    sk_metrics_df = pd.DataFrame(
        {
            "LR_mae": [sk["lr_metrics"]["mae"]],
            "LR_mse": [sk["lr_metrics"]["mse"]],
            "QR_mae": [sk["qr_metrics"]["mae"]],
            "QR_mse": [sk["qr_metrics"]["mse"]],
        }
    )
    xo_metrics_df = pd.DataFrame(
        {
            "LR_mae": [xo["lr_metrics"]["mae"]],
            "LR_mse": [xo["lr_metrics"]["mse"]],
            "QR_mae": [xo["qr_metrics"]["mae"]],
            "QR_mse": [xo["qr_metrics"]["mse"]],
        }
    )
    return [
        ("LR vs QR metrics", sk_metrics_df, xo_metrics_df),
    ]


# ---------------------------------------------------------------------------
# Plot: defined here — reader sees figure creation and compositing
# ---------------------------------------------------------------------------


def plot(sk, xo, comparator):
    """Build quantile regression plots and composite figure."""
    target_col = comparator.target
    dataset_type = comparator.name.split("_")[-1]  # "normal" or "pareto"

    # Build sklearn plot dataframe
    sk_plot_df = comparator.df.copy()
    sk_plot_df["pred_q05"] = sk["QR_05"]["predictions"]
    sk_plot_df["pred_q50"] = sk["QR_50"]["predictions"]
    sk_plot_df["pred_q95"] = sk["QR_95"]["predictions"]
    sk_plot_df["out_bounds"] = sk["out_bounds"].astype(int)

    # Build xorq plot dataframe
    xo_plot_df = comparator.df.copy()
    xo_plot_df["pred_q05"] = xo["QR_05"]["predictions"]
    xo_plot_df["pred_q50"] = xo["QR_50"]["predictions"]
    xo_plot_df["pred_q95"] = xo["QR_95"]["predictions"]
    xo_plot_df["out_bounds"] = xo["out_bounds"].astype(int)

    sk_fig = _build_quantile_plot(sk_plot_df, dataset_type, target_col)
    xo_fig = _build_quantile_plot(xo_plot_df, dataset_type, target_col)

    # Composite: sklearn (left) | xorq (right)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].axis("off")

    fig.suptitle(
        f"Quantile Regression ({dataset_type}): sklearn vs xorq",
        fontsize=16,
    )
    fig.tight_layout()
    os.makedirs("imgs", exist_ok=True)
    out = f"imgs/quantile_regression_{dataset_type}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out}")


# ---------------------------------------------------------------------------
# Shared sklearn pipelines
# ---------------------------------------------------------------------------

SHARED_SKLEARN_PIPELINES = (
    ("QR_05", SklearnPipeline([("qr", QuantileRegressor(quantile=0.05, alpha=0))])),
    ("QR_50", SklearnPipeline([("qr", QuantileRegressor(quantile=0.5, alpha=0))])),
    ("QR_95", SklearnPipeline([("qr", QuantileRegressor(quantile=0.95, alpha=0))])),
    ("LR", SklearnPipeline([("lr", LinearRegression())])),
    ("QR_median", SklearnPipeline([("qr", QuantileRegressor(quantile=0.5, alpha=0))])),
)

_df = _load_data()


# ---------------------------------------------------------------------------
# Comparators: one per dataset type
# ---------------------------------------------------------------------------

comparators = {
    "normal": SklearnXorqComparator(
        name="quantile_normal",
        named_pipelines=SHARED_SKLEARN_PIPELINES,
        df=_df,
        features=FEATURE_COLS,
        target="y_normal",
        pred_col=PRED_COL,
        metrics=(("mae", mean_absolute_error), ("mse", mean_squared_error)),
        build_exprs_fn=build_exprs_with_metrics("y_normal"),
        compute_sklearn_fn=compute_sklearn,
        compute_xorq_fn=compute_xorq,
        build_assertions_fn=build_assertions,
        plot_fn=plot,
    ),
    "pareto": SklearnXorqComparator(
        name="quantile_pareto",
        named_pipelines=SHARED_SKLEARN_PIPELINES,
        df=_df,
        features=FEATURE_COLS,
        target="y_pareto",
        pred_col=PRED_COL,
        metrics=(("mae", mean_absolute_error), ("mse", mean_squared_error)),
        build_exprs_fn=build_exprs_with_metrics("y_pareto"),
        compute_sklearn_fn=compute_sklearn,
        compute_xorq_fn=compute_xorq,
        build_assertions_fn=build_assertions,
        plot_fn=plot,
    ),
}

# ── Module-level exprs (for xorq build --expr) ────────────────
xorq_normal_exprs = comparators["normal"].deferred_exprs
xorq_pareto_exprs = comparators["pareto"].deferred_exprs

xorq_normal_qr05_fitted_pipeline = xorq_normal_exprs["QR_05"]["fitted_pipeline"]
xorq_normal_qr05_preds = xorq_normal_exprs["QR_05"]["preds"]
xorq_normal_qr05_metrics = xorq_normal_exprs["QR_05"]["metrics"]
xorq_normal_qr50_fitted_pipeline = xorq_normal_exprs["QR_50"]["fitted_pipeline"]
xorq_normal_qr50_preds = xorq_normal_exprs["QR_50"]["preds"]
xorq_normal_qr50_metrics = xorq_normal_exprs["QR_50"]["metrics"]
xorq_normal_qr95_fitted_pipeline = xorq_normal_exprs["QR_95"]["fitted_pipeline"]
xorq_normal_qr95_preds = xorq_normal_exprs["QR_95"]["preds"]
xorq_normal_qr95_metrics = xorq_normal_exprs["QR_95"]["metrics"]
xorq_normal_lr_fitted_pipeline = xorq_normal_exprs["LR"]["fitted_pipeline"]
xorq_normal_lr_preds = xorq_normal_exprs["LR"]["preds"]
xorq_normal_lr_metrics = xorq_normal_exprs["LR"]["metrics"]
xorq_normal_qr_median_fitted_pipeline = xorq_normal_exprs["QR_median"][
    "fitted_pipeline"
]
xorq_normal_qr_median_preds = xorq_normal_exprs["QR_median"]["preds"]
xorq_normal_qr_median_metrics = xorq_normal_exprs["QR_median"]["metrics"]

xorq_pareto_qr05_fitted_pipeline = xorq_pareto_exprs["QR_05"]["fitted_pipeline"]
xorq_pareto_qr05_preds = xorq_pareto_exprs["QR_05"]["preds"]
xorq_pareto_qr05_metrics = xorq_pareto_exprs["QR_05"]["metrics"]
xorq_pareto_qr50_fitted_pipeline = xorq_pareto_exprs["QR_50"]["fitted_pipeline"]
xorq_pareto_qr50_preds = xorq_pareto_exprs["QR_50"]["preds"]
xorq_pareto_qr50_metrics = xorq_pareto_exprs["QR_50"]["metrics"]
xorq_pareto_qr95_fitted_pipeline = xorq_pareto_exprs["QR_95"]["fitted_pipeline"]
xorq_pareto_qr95_preds = xorq_pareto_exprs["QR_95"]["preds"]
xorq_pareto_qr95_metrics = xorq_pareto_exprs["QR_95"]["metrics"]
xorq_pareto_lr_fitted_pipeline = xorq_pareto_exprs["LR"]["fitted_pipeline"]
xorq_pareto_lr_preds = xorq_pareto_exprs["LR"]["preds"]
xorq_pareto_lr_metrics = xorq_pareto_exprs["LR"]["metrics"]
xorq_pareto_qr_median_fitted_pipeline = xorq_pareto_exprs["QR_median"][
    "fitted_pipeline"
]
xorq_pareto_qr_median_preds = xorq_pareto_exprs["QR_median"]["preds"]
xorq_pareto_qr_median_metrics = xorq_pareto_exprs["QR_median"]["metrics"]


# =========================================================================
# Run
# =========================================================================


def main():
    for dataset_type, comparator in comparators.items():
        print(f"\n=== DATASET: {dataset_type.upper()} ===")

        print("=== SKLEARN WAY ===")
        sk = comparator.compute_with_sklearn()

        print("\n=== XORQ WAY ===")
        xo = comparator.compute_with_xorq()

        comparator.assert_values(sk, xo)
        comparator.plot(sk, xo)


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
