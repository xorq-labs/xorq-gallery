"""Time-related feature engineering
====================================

sklearn: Build a ColumnTransformer with periodic SplineTransformer for cyclic
time features + HistGradientBoostingRegressor, fit eagerly on numpy arrays,
evaluate MAE. Split with train_test_split(shuffle=False) to preserve time order.

xorq: Same pipeline wrapped in Pipeline.from_instance. Data is an ibis
expression, split via deferred_cross_val_score with TimeSeriesSplit, fit/predict
deferred, metrics via deferred_sklearn_metric.

Both splits preserve temporal order and produce identical train/test rows --
sklearn via shuffle=False, xorq via TimeSeriesSplit fold assignment.

Dataset: Bike Sharing Demand (OpenML)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xorq.api as xo
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, SplineTransformer
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    deferred_sequential_split,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

y_col = "count"
time_features = ["hour", "weekday", "month"]
weather_col = "weather"
numerical_weather = ["temp", "feel_temp", "humidity", "windspeed"]
all_feature_cols = time_features + [weather_col] + numerical_weather


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    bike_sharing = fetch_openml(
        "Bike_Sharing_Demand", version=2, as_frame=True, parser="pandas"
    )
    df = bike_sharing.frame

    # Collapse rare category
    df["weather"] = (
        df["weather"]
        .astype(object)
        .replace(to_replace="heavy_rain", value="rain")
        .astype("category")
    )

    # Ensure numeric types
    for col in ("hour", "weekday", "month"):
        df[col] = df[col].astype(int)
    for col in ("temp", "feel_temp", "humidity", "windspeed"):
        df[col] = df[col].astype(float)

    # Target: fraction of max demand
    df[y_col] = df[y_col] / df[y_col].max()

    # Row index for temporal ordering
    df["row_idx"] = range(len(df))

    return df


# ---------------------------------------------------------------------------
# Plotting helpers (used by deferred_matplotlib_plot)
# ---------------------------------------------------------------------------

N_PLOT = 96


def _build_prediction_figure(title):
    """Return a UDAF-compatible plotting function for actual vs predicted."""
    def _plot(df):
        actual = df[y_col].values
        predicted = df["pred"].values
        mae = mean_absolute_error(actual, predicted)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            range(N_PLOT), actual[-N_PLOT:],
            color="black", linewidth=1.2, label="actual",
        )
        ax.plot(
            range(N_PLOT), predicted[-N_PLOT:],
            alpha=0.8, label=f"predicted (MAE={mae:.4f})",
        )
        ax.set_title(title)
        ax.set_xlabel("Hours")
        ax.set_ylabel("Fraction of max demand")
        ax.legend(fontsize=8)
        plt.tight_layout()
        return fig

    return _plot


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
            ("hgbr", HistGradientBoostingRegressor(max_iter=200, random_state=42)),
        ]
    )

    return sklearn_spline_ridge, sklearn_hgbr


# =========================================================================
# SKLEARN WAY -- eager, train_test_split(shuffle=False)
# =========================================================================


def sklearn_way(df, sklearn_spline_ridge, sklearn_hgbr):
    """Eager sklearn: time-ordered split, fit, predict, score."""
    X = df[all_feature_cols]
    y = df[y_col]

    # shuffle=False preserves temporal order: first rows train, last rows test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3333, shuffle=False
    )

    # Spline+Ridge
    sklearn_spline_ridge.fit(X_train, y_train)
    spline_pred = sklearn_spline_ridge.predict(X_test)
    spline_mae = mean_absolute_error(y_test, spline_pred)
    print(f"  sklearn Spline+Ridge: MAE = {spline_mae:.4f}")

    # HGBR
    sklearn_hgbr.fit(X_train, y_train)
    hgbr_pred = sklearn_hgbr.predict(X_test)
    hgbr_mae = mean_absolute_error(y_test, hgbr_pred)
    print(f"  sklearn HGBR: MAE = {hgbr_mae:.4f}")

    return {
        "Spline+Ridge": (spline_mae, (y_test.values, spline_pred)),
        "HGBR": (hgbr_mae, (y_test.values, hgbr_pred)),
    }


# =========================================================================
# XORQ WAY -- deferred, TimeSeriesSplit(n_splits=2) fold_1
# =========================================================================


def xorq_way(df, sklearn_spline_ridge, sklearn_hgbr):
    """Deferred xorq: TimeSeriesSplit fold assignment, Pipeline.from_instance.

    Returns deferred predictions and metrics for both models.
    Nothing is executed until ``.execute()``.
    """
    con = xo.connect()
    data = con.register(df, "bike_sharing")
    features = tuple(all_feature_cols)

    train_data, test_data = deferred_sequential_split(
        data, features=features, target=y_col, order_by="row_idx"
    )

    make_metric = deferred_sklearn_metric(target=y_col, pred="pred")

    # Spline+Ridge
    spline_pipe = Pipeline.from_instance(sklearn_spline_ridge)
    spline_fitted = spline_pipe.fit(train_data, features=features, target=y_col)
    spline_preds = spline_fitted.predict(test_data, name="pred")
    spline_metrics = spline_preds.agg(mae=make_metric(metric=mean_absolute_error))

    # HGBR
    hgbr_pipe = Pipeline.from_instance(sklearn_hgbr)
    hgbr_fitted = hgbr_pipe.fit(train_data, features=features, target=y_col)
    hgbr_preds = hgbr_fitted.predict(test_data, name="pred")
    hgbr_metrics = hgbr_preds.agg(mae=make_metric(metric=mean_absolute_error))

    return {
        "Spline+Ridge": (spline_preds, spline_metrics),
        "HGBR": (hgbr_preds, hgbr_metrics),
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()
    sklearn_spline_ridge, sklearn_hgbr = _build_pipelines()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(df, sklearn_spline_ridge, sklearn_hgbr)

    print("\n=== XORQ WAY ===")
    deferred = xorq_way(df, sklearn_spline_ridge, sklearn_hgbr)

    # Execute deferred expressions and print metrics
    xo_plots = {}
    for name, (preds_expr, metrics_expr) in deferred.items():
        metrics_df = metrics_expr.execute()
        mae = metrics_df["mae"].iloc[0]
        print(f"  xorq   {name}: MAE = {mae:.4f}")
        xo_plots[name] = deferred_matplotlib_plot(
            preds_expr, _build_prediction_figure(f"xorq - {name}")
        ).execute()

    # Build sklearn subplot figures natively
    model_names = ["Spline+Ridge", "HGBR"]

    sk_figs = {}
    for name in model_names:
        mae, (y_test, y_pred) = sk_results[name]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            range(N_PLOT), y_test[-N_PLOT:],
            color="black", linewidth=1.2, label="actual",
        )
        ax.plot(
            range(N_PLOT), y_pred[-N_PLOT:],
            alpha=0.8, label=f"predicted (MAE={mae:.4f})",
        )
        ax.set_title(f"sklearn - {name}")
        ax.set_xlabel("Hours")
        ax.set_ylabel("Fraction of max demand")
        ax.legend(fontsize=8)
        plt.tight_layout()
        sk_figs[name] = fig

    # Composite: 2x2 grid -- sklearn (left) | xorq deferred (right)
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    for row, name in enumerate(model_names):
        # sklearn subplot (rendered to image)
        axes[row, 0].imshow(fig_to_image(sk_figs[name]))
        axes[row, 0].axis("off")

        # xorq subplot (loaded from deferred PNG bytes)
        xo_img = load_plot_bytes(xo_plots[name])
        axes[row, 1].imshow(xo_img)
        axes[row, 1].axis("off")

    plt.suptitle("Cyclical Feature Engineering: sklearn vs xorq", fontsize=14)
    plt.tight_layout()
    out = "imgs/cyclical_feature_engineering.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
