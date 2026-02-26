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
"""

from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, SplineTransformer
from xorq.expr.ml.cross_validation import deferred_cross_val_score
from xorq.expr.ml.pipeline_lib import Pipeline

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

name_to_pipeline = {
    SPLINE_RIDGE: SklearnPipeline(
        [
            ("preprocess", _spline_preprocessor),
            ("ridge", RidgeCV(alphas=tuple(np.logspace(-6, 6, 25).tolist()))),
        ]
    ),
    HGBR: SklearnPipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            [weather_col],
                        ),
                        ("num", MinMaxScaler(), numerical_weather),
                        ("time", "passthrough", time_features),
                    ]
                ),
            ),
            ("hgbr", HistGradientBoostingRegressor(max_iter=200, random_state=42)),
        ]
    ),
}


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


# =========================================================================
# SKLEARN WAY -- eager cross_validate with TimeSeriesSplit
# =========================================================================


def compute_with_sklearn(name_to_pipeline, df):
    """Eager sklearn: cross-validate each pipeline with TimeSeriesSplit.

    Returns
    -------
    dict
        Keys: method names, values: per-fold MAE arrays
    """
    X = df[list(all_feature_cols)]
    y = df[y_col]

    results = {
        name: -cross_validate(
            pipe,
            X,
            y,
            cv=CV_SPLITTER,
            scoring="neg_mean_absolute_error",
        )["test_score"]
        for name, pipe in name_to_pipeline.items()
    }
    for name, mae_scores in results.items():
        print(
            f"  sklearn {name:15s}: MAE={mae_scores.mean():.4f} (+/-{mae_scores.std():.4f})"
        )
    return results


# =========================================================================
# XORQ WAY -- deferred cross-validation with TimeSeriesSplit
# =========================================================================


def compute_with_xorq(name_to_xorq_cv):
    """Execute deferred CV scores for each pipeline.

    Returns
    -------
    dict
        Keys: method names, values: per-fold MAE arrays
    """
    xo_scores = {name: -cv.execute() for name, cv in name_to_xorq_cv.items()}
    for name, mae in xo_scores.items():
        print(f"  xorq   {name:15s}: MAE={mae.mean():.4f} (+/-{mae.std():.4f})")
    return xo_scores


con = xo.connect()
data = con.register(_load_data(), "bike_sharing")

name_to_xorq_cv = {
    name: deferred_cross_val_score(
        Pipeline.from_instance(pipeline),
        data,
        all_feature_cols,
        y_col,
        cv=CV_SPLITTER,
        order_by=ROW_IDX,
        scoring="neg_mean_absolute_error",
    )
    for name, pipeline in name_to_pipeline.items()
}
(xorq_spline_ridge_cv, xorq_hgbr_cv) = (name_to_xorq_cv[name] for name in methods)


# =========================================================================
# Run and plot side by side
# =========================================================================


def compare_results(sk_scores, xo_scores):
    print("\n=== ASSERTIONS ===")
    sk_df = pd.DataFrame(sk_scores)
    xo_df = pd.DataFrame(xo_scores)
    pd.testing.assert_frame_equal(sk_df, xo_df, rtol=1e-6)
    print("Per-fold MAE scores match for all models.")
    print("Assertions passed.")


def save_comparison_plot(sk_scores, xo_scores):
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
    save_fig("imgs/cyclical_feature_engineering.png", fig, bbox_inches="tight")


def main():
    df = _load_data()

    print("=== SKLEARN WAY ===")
    sk_scores = compute_with_sklearn(name_to_pipeline, df)

    print("\n=== XORQ WAY ===")
    xo_scores = compute_with_xorq(name_to_xorq_cv)

    compare_results(sk_scores, xo_scores)
    save_comparison_plot(sk_scores, xo_scores)


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
