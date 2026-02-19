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

from xorq_gallery.utils import deferred_sequential_split


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------

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
y_col = "count"
df[y_col] = df[y_col] / df[y_col].max()

# Row index for temporal ordering
df["row_idx"] = range(len(df))

# Feature groups
time_features = ["hour", "weekday", "month"]
weather_col = "weather"
numerical_weather = ["temp", "feel_temp", "humidity", "windspeed"]
all_feature_cols = time_features + [weather_col] + numerical_weather

# ---------------------------------------------------------------------------
# Shared pipeline definitions
# ---------------------------------------------------------------------------

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


# =========================================================================
# SKLEARN WAY -- eager, train_test_split(shuffle=False)
# =========================================================================


def sklearn_way(df):
    """Eager sklearn: time-ordered split, fit, predict, score."""
    X = df[all_feature_cols]
    y = df[y_col]

    # shuffle=False preserves temporal order: first rows train, last rows test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3333, shuffle=False
    )

    results = {}
    for name, pipe in [
        ("Spline+Ridge", sklearn_spline_ridge),
        ("HGBR", sklearn_hgbr),
    ]:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {
            "mae": mae,
            "y_test": y_test.values,
            "y_pred": y_pred,
        }
        print(f"  sklearn {name}: MAE = {mae:.4f}")

    return results


# =========================================================================
# XORQ WAY -- deferred, TimeSeriesSplit(n_splits=2) fold_1
# =========================================================================


def xorq_way(df):
    """Deferred xorq: TimeSeriesSplit fold assignment, Pipeline.from_instance."""
    con = xo.connect()
    data = con.register(df, "bike_sharing")
    features = tuple(all_feature_cols)

    train_data, test_data = deferred_sequential_split(
        data, features=features, target=y_col, order_by="row_idx"
    )

    results = {}
    for name, sk_pipe in [
        ("Spline+Ridge", sklearn_spline_ridge),
        ("HGBR", sklearn_hgbr),
    ]:
        xorq_pipe = Pipeline.from_instance(sk_pipe)
        fitted = xorq_pipe.fit(train_data, features=features, target=y_col)
        preds = fitted.predict(test_data, name="pred")

        mae_expr = preds.agg(
            mae=deferred_sklearn_metric(
                target=y_col, pred="pred", metric=mean_absolute_error
            ),
        )

        mae = mae_expr.execute()["mae"].iloc[0]
        preds_df = preds.execute()
        results[name] = {
            "mae": mae,
            "y_test": preds_df[y_col].values,
            "y_pred": preds_df["pred"].values,
        }
        print(f"  xorq   {name}: MAE = {mae:.4f}")

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================

if __name__ in ("__main__", "__pytest_main__"):
    os.makedirs("imgs", exist_ok=True)

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(df)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(df)

    # Side-by-side plot: last 96 hours of predictions
    n_plot = 96
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True, sharey=True)

    for col, (label, results) in enumerate(
        [("sklearn", sk_results), ("xorq", xo_results)]
    ):
        for row, name in enumerate(["Spline+Ridge", "HGBR"]):
            ax = axes[row, col]
            r = results[name]
            ax.plot(
                range(n_plot),
                r["y_test"][-n_plot:],
                color="black",
                linewidth=1.2,
                label="actual",
            )
            ax.plot(
                range(n_plot),
                r["y_pred"][-n_plot:],
                alpha=0.8,
                label=f"predicted (MAE={r['mae']:.4f})",
            )
            ax.set_title(f"{label} - {name}")
            ax.legend(fontsize=8)
            if col == 0:
                ax.set_ylabel("Fraction of max demand")
            if row == 1:
                ax.set_xlabel("Hours")

    plt.suptitle("Cyclical Feature Engineering: sklearn vs xorq", fontsize=14)
    plt.tight_layout()
    plt.savefig("imgs/cyclical_feature_engineering.png", dpi=150)
    plt.close()

    pytest_examples_passed = True
