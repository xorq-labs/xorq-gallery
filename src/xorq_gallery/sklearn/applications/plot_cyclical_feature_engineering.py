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

import os

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

from xorq_gallery.utils import fig_to_image


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


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    bike_sharing = fetch_openml(
        "Bike_Sharing_Demand", version=2, as_frame=True, parser="pandas"
    )
    df = bike_sharing.frame
    return (
        df
        .assign(**{
            weather_col: (
                df[weather_col]
                .astype(object)
                .replace(to_replace="heavy_rain", value="rain")
                .astype("category")
            ),
            y_col: df[y_col].div(df[y_col].max()),
            ROW_IDX: range(len(df)),
        })
        .astype({col: int for col in ("hour", "weekday", "month")})
        .astype({col: float for col in ("temp", "feel_temp", "humidity", "windspeed")})
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

MODEL_NAMES = ("Spline+Ridge", "HGBR")


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
# Shared pipeline definitions
# ---------------------------------------------------------------------------


def _build_pipelines():
    spline_preprocessor = ColumnTransformer(
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

    sklearn_spline_ridge = SklearnPipeline(
        [
            ("preprocess", spline_preprocessor),
            ("ridge", RidgeCV(alphas=tuple(np.logspace(-6, 6, 25).tolist()))),
        ]
    )

    sklearn_hgbr = SklearnPipeline(
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
    )

    return sklearn_spline_ridge, sklearn_hgbr


# =========================================================================
# SKLEARN WAY -- eager cross_validate with TimeSeriesSplit
# =========================================================================


def sklearn_way(df, pipelines):
    """Eager sklearn: cross-validate each pipeline with TimeSeriesSplit.

    df must be sorted by ROW_IDX so fold assignments match the xorq side.
    """
    X = df[list(all_feature_cols)]
    y = df[y_col]

    results = {}
    for name, pipe in zip(MODEL_NAMES, pipelines):
        result = cross_validate(
            pipe, X, y,
            cv=CV_SPLITTER,
            scoring="neg_mean_absolute_error",
        )
        mae_scores = -result["test_score"]
        results[name] = mae_scores
        print(f"  sklearn {name:15s}: MAE={mae_scores.mean():.4f} (+/-{mae_scores.std():.4f})")

    return results


# =========================================================================
# XORQ WAY -- deferred cross-validation with TimeSeriesSplit
# =========================================================================


def xorq_way(data, pipelines):
    """Deferred xorq: deferred_cross_val_score per pipeline with TimeSeriesSplit.

    Returns dict of CrossValScore objects keyed by model name.
    Nothing is executed until .execute().
    """
    results = {}
    for name, sk_pipe in zip(MODEL_NAMES, pipelines):
        xorq_pipe = Pipeline.from_instance(sk_pipe)
        cv_result = deferred_cross_val_score(
            xorq_pipe, data, all_feature_cols, y_col,
            cv=CV_SPLITTER, order_by=ROW_IDX,
            scoring="neg_mean_absolute_error",
        )
        results[name] = cv_result

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()
    # Sort by ROW_IDX so sklearn TimeSeriesSplit sees the same row order as xorq
    df = df.sort_values(ROW_IDX).reset_index(drop=True)

    con = xo.connect()
    table = con.register(df, "bike_sharing")

    print("=== SKLEARN WAY ===")
    sk_pipelines = _build_pipelines()
    sk_results = sklearn_way(df, sk_pipelines)

    print("\n=== XORQ WAY ===")
    xo_pipelines = _build_pipelines()
    xo_deferred = xorq_way(table, xo_pipelines)

    # Execute deferred CV scores
    xo_scores = {
        name: -xo_deferred[name].execute()
        for name in MODEL_NAMES
    }
    for name in MODEL_NAMES:
        mae = xo_scores[name]
        print(f"  xorq   {name:15s}: MAE={mae.mean():.4f} (+/-{mae.std():.4f})")

    # Assert: per-fold scores match via DataFrame comparison
    print("\n=== ASSERTIONS ===")
    sk_df = pd.DataFrame(sk_results)
    xo_df = pd.DataFrame(xo_scores)
    pd.testing.assert_frame_equal(sk_df, xo_df, rtol=1e-6)
    print("Per-fold MAE scores match for all models.")
    print("Assertions passed.")

    # Build plots
    sk_fig = _plot_cv_results(sk_results, "sklearn - Cyclical Feature Engineering (MAE)")
    xo_fig = _plot_cv_results(xo_scores, "xorq - Cyclical Feature Engineering (MAE)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle("Cyclical Feature Engineering: sklearn vs xorq", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout()
    out = "imgs/cyclical_feature_engineering.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
