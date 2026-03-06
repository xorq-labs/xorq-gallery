"""Time-related feature engineering
====================================

sklearn: Build a ColumnTransformer with periodic SplineTransformer for cyclic
time features + HistGradientBoostingRegressor, evaluate using cross_validate
with TimeSeriesSplit to preserve temporal order. Compare MAE across
Spline+Ridge and HGBR models.

xorq: Same pipelines wrapped in Pipeline.from_instance, deferred
cross-validation via deferred_cross_val_score with TimeSeriesSplit and
order_by=ROW_IDX. Per-fold MAE scores match sklearn exactly.

Both produce identical cross-validation scores.

Dataset: Bike Sharing Demand (OpenML)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/applications/plot_cyclical_feature_engineering.py
"""

from __future__ import annotations

from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, SplineTransformer
from xorq.expr.ml.cross_validation import deferred_cross_val_score
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    split_data_nop,
)
from xorq_gallery.utils import fig_to_image, save_fig


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

y_col = "count"
ROW_IDX = "row_idx"
time_features = ("hour", "weekday", "month")
weather_col = "weather"
numerical_weather = ("temp", "feel_temp", "humidity", "windspeed")
all_feature_cols = time_features + (weather_col,) + numerical_weather
RANDOM_STATE = 42
N_SPLITS = 5

CV_SPLITTER = TimeSeriesSplit(n_splits=N_SPLITS)

methods = (SPLINE_RIDGE, HGBR) = ("Spline+Ridge", "HGBR")

_spline_preprocessor = ColumnTransformer(
    [
        (
            "hour_spline",
            SplineTransformer(
                degree=3, n_knots=12, knots="uniform", extrapolation="periodic"
            ),
            ["hour"],
        ),
        (
            "weekday_spline",
            SplineTransformer(
                degree=3, n_knots=7, knots="uniform", extrapolation="periodic"
            ),
            ["weekday"],
        ),
        (
            "month_spline",
            SplineTransformer(
                degree=3, n_knots=12, knots="uniform", extrapolation="periodic"
            ),
            ["month"],
        ),
        (
            "weather_ohe",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            [weather_col],
        ),
        ("num_scale", MinMaxScaler(), numerical_weather),
    ]
)

names_pipelines = (
    (
        SPLINE_RIDGE,
        SklearnPipeline(
            [
                ("preprocess", _spline_preprocessor),
                ("ridge", RidgeCV(alphas=tuple(np.logspace(-6, 6, 25).tolist()))),
            ]
        ),
    ),
    (
        HGBR,
        SklearnPipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(
                        [
                            (
                                "cat",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                                [weather_col],
                            ),
                            ("num", MinMaxScaler(), numerical_weather),
                            ("time", "passthrough", time_features),
                        ]
                    ),
                ),
                (
                    "hgbr",
                    HistGradientBoostingRegressor(max_iter=200, random_state=42),
                ),
            ]
        ),
    ),
)


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


@cache
def _load_data():
    bike_sharing = fetch_openml(
        "Bike_Sharing_Demand", version=2, as_frame=True, parser="pandas"
    )
    df = bike_sharing.frame
    return (
        df.assign(
            **{
                weather_col: (
                    df[weather_col]
                    .astype(object)
                    .replace(to_replace="heavy_rain", value="rain")
                    .astype("category")
                ),
                y_col: df[y_col].div(df[y_col].max()),
                ROW_IDX: range(len(df)),
            }
        )
        .astype({col: int for col in ("hour", "weekday", "month")})
        .astype({col: float for col in ("temp", "feel_temp", "humidity", "windspeed")})
        .sort_values(ROW_IDX)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_cv_results(results, title):
    """Bar plot comparing MAE across models with per-fold error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(results.keys())
    means = [np.mean(results[n]) for n in names]
    stds = [np.std(results[n]) for n in names]

    ax.bar(names, means, yerr=stds, capsize=5, color=["C0", "C1"])
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(title)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.002, f"{m:.4f}", ha="center", fontsize=10)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Custom make_*_result for cross-validation
# ---------------------------------------------------------------------------


def _make_sklearn_cv_result(
    pipeline, train_data, test_data, features, target, metrics_names_funcs
):
    """Sklearn: cross-validate with TimeSeriesSplit, return per-fold MAE."""
    X = train_data[list(features)]
    y = train_data[target]

    cv_result = cross_validate(
        clone(pipeline),
        X,
        y,
        cv=CV_SPLITTER,
        scoring="neg_mean_absolute_error",
    )
    mae_scores = -cv_result["test_score"]
    return {
        "fitted": None,
        "preds": mae_scores,
        "metrics": {"mae_mean": mae_scores.mean(), "mae_std": mae_scores.std()},
    }


def _make_deferred_xorq_cv_result(
    pipeline, train_data, test_data, features, target, metrics_names_funcs, pred
):
    """Deferred xorq: deferred_cross_val_score with TimeSeriesSplit."""
    xorq_pipe = Pipeline.from_instance(pipeline)
    cv_result = deferred_cross_val_score(
        xorq_pipe,
        train_data,
        features,
        target,
        cv=CV_SPLITTER,
        order_by=ROW_IDX,
        scoring="neg_mean_absolute_error",
    )
    return {
        "xorq_fitted": None,
        "preds": cv_result,
        "metrics": {},
        "other": {},
    }


def _make_xorq_cv_result(deferred_xorq_result):
    """Materialize deferred CV scores."""
    scores = deferred_xorq_result["preds"].execute()
    mae_scores = -scores
    return {
        "fitted": None,
        "preds": mae_scores,
        "metrics": {"mae_mean": mae_scores.mean(), "mae_std": mae_scores.std()},
    }


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def compare_results(comparator):
    assert sorted(sklearn_results := comparator.sklearn_results) == sorted(
        xorq_results := comparator.xorq_results
    )
    print("\n=== Comparing Results ===")
    for name in sklearn_results:
        sk_mae = sklearn_results[name]["preds"]
        xo_mae = xorq_results[name]["preds"]
        print(
            f"  {name:15s} MAE - sklearn: {sk_mae.mean():.4f} (+/-{sk_mae.std():.4f})"
            f", xorq: {xo_mae.mean():.4f} (+/-{xo_mae.std():.4f})"
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame({"mae": sk_mae}),
            pd.DataFrame({"mae": xo_mae}),
            rtol=1e-6,
        )
    print("Assertions passed: per-fold MAE scores match.")


def plot_results(comparator):
    sk_scores = {name: comparator.sklearn_results[name]["preds"] for name in methods}
    xo_scores = {name: comparator.xorq_results[name]["preds"] for name in methods}

    sk_fig = _plot_cv_results(sk_scores, "sklearn - Cyclical Feature Engineering (MAE)")
    xo_fig = _plot_cv_results(xo_scores, "xorq - Cyclical Feature Engineering (MAE)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(
        "Cyclical Feature Engineering: sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()

    plt.close(sk_fig)
    plt.close(xo_fig)
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=all_feature_cols,
    target=y_col,
    pred="pred",
    metrics_names_funcs=(),
    load_data=_load_data,
    split_data=split_data_nop,
    make_sklearn_result=_make_sklearn_cv_result,
    make_deferred_xorq_result=_make_deferred_xorq_cv_result,
    make_xorq_result=_make_xorq_cv_result,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)

# Module-level deferred exprs
(xorq_spline_ridge_cv, xorq_hgbr_cv) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    comparator.result_comparison
    save_fig("imgs/cyclical_feature_engineering.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
