"""Lagged features for time series forecasting
================================================

sklearn: Engineer lagged features with pandas shift, fit
HistGradientBoostingRegressor on numpy arrays, evaluate MAE/RMSE.
Split with train_test_split(shuffle=False) to preserve time order.

xorq: Engineer lagged features with ibis window functions (.lag()),
split via TimeSeriesSplit fold assignment, wrap the same model in
Pipeline.from_instance, fit/predict deferred, evaluate with
deferred_sklearn_metric.

Both splits preserve temporal order and produce identical train/test
rows independently.

Dataset: Bike Sharing Demand (OpenML)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import xorq.api as xo
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    deferred_sequential_split,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

target = "count"
calendar_features = ["hour", "weekday", "month"]
weather_features = ["temp", "humidity", "windspeed"]
model_params = dict(max_iter=200, max_depth=8, learning_rate=0.1, random_state=42)


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    bike_sharing = fetch_openml(
        "Bike_Sharing_Demand", version=2, as_frame=True, parser="pandas"
    )
    df = bike_sharing.frame

    df["count"] = df["count"].astype(float)
    df["hour"] = df["hour"].astype(int)
    df["weekday"] = df["weekday"].astype(int)
    df["month"] = df["month"].astype(int)
    df["temp"] = df["temp"].astype(float)
    df["humidity"] = df["humidity"].astype(float)
    df["windspeed"] = df["windspeed"].astype(float)

    # Row index for temporal ordering
    df["row_idx"] = range(len(df))

    return df


# ---------------------------------------------------------------------------
# Plotting helpers (used by deferred_matplotlib_plot)
# ---------------------------------------------------------------------------

N_PLOT = 96


def _build_xorq_figure(df):
    """Plot xorq predictions vs actuals for the last N_PLOT hours."""
    actual = df[target].values
    predicted = df["pred"].values
    mae = mean_absolute_error(actual, predicted)

    fig, ax = plt.subplots(figsize=(8, 5))
    hours = range(N_PLOT)
    ax.plot(hours, actual[-N_PLOT:], color="black", linewidth=1.2, label="actual")
    ax.plot(
        hours,
        predicted[-N_PLOT:],
        color="tab:blue",
        linewidth=1,
        linestyle="--",
        label=f"predicted (MAE={mae:.2f})",
    )
    ax.set_title("xorq - HGBR with lagged features")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Bike count")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- lagged features via pandas, eager split
# =========================================================================


def sklearn_way(df):
    """Eager sklearn: pandas shift for lags, shuffle=False split."""
    # Engineer lagged features with pandas
    ldf = df[["count", "row_idx"] + calendar_features + weather_features].copy()
    ldf["lagged_count_1h"] = ldf["count"].shift(1)
    ldf["lagged_count_2h"] = ldf["count"].shift(2)
    ldf["lagged_count_3h"] = ldf["count"].shift(3)
    ldf["lagged_count_1d"] = ldf["count"].shift(24)
    ldf["lagged_count_7d"] = ldf["count"].shift(24 * 7)
    ldf = ldf.dropna()

    lagged_cols = [c for c in ldf.columns if c.startswith("lagged_")]
    feature_cols = lagged_cols + calendar_features + weather_features

    X = ldf[feature_cols]
    y = ldf[target]

    # shuffle=False preserves temporal order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3333, shuffle=False
    )

    model = HistGradientBoostingRegressor(**model_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  sklearn: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

    return {"mae": mae, "rmse": rmse, "y_test": y_test.values, "y_pred": y_pred}


# =========================================================================
# XORQ WAY -- lagged features via ibis, deferred TimeSeriesSplit
# =========================================================================


def xorq_way(df):
    """Deferred xorq: ibis .lag(), TimeSeriesSplit fold, Pipeline.from_instance.

    Returns deferred expressions for the plot (PNG bytes via UDAF) and
    metrics.  Nothing is executed until the caller calls ``.execute()``.
    """
    con = xo.connect()
    data = con.register(df, "bike_sharing")

    # Lagged features via ibis window functions
    data_with_lags = data.mutate(
        lagged_count_1h=data[target].lag(1).over(order_by="row_idx"),
        lagged_count_2h=data[target].lag(2).over(order_by="row_idx"),
        lagged_count_3h=data[target].lag(3).over(order_by="row_idx"),
        lagged_count_1d=data[target].lag(24).over(order_by="row_idx"),
        lagged_count_7d=data[target].lag(24 * 7).over(order_by="row_idx"),
    ).drop_null(
        subset=[
            "lagged_count_1h",
            "lagged_count_2h",
            "lagged_count_3h",
            "lagged_count_1d",
            "lagged_count_7d",
        ]
    )

    lagged_features = [
        "lagged_count_1h",
        "lagged_count_2h",
        "lagged_count_3h",
        "lagged_count_1d",
        "lagged_count_7d",
    ]
    all_features = lagged_features + calendar_features + weather_features

    train_data, test_data = deferred_sequential_split(
        data_with_lags, features=tuple(all_features), target=target, order_by="row_idx"
    )

    sk_pipe = SklearnPipeline([("hgbr", HistGradientBoostingRegressor(**model_params))])
    xorq_pipe = Pipeline.from_instance(sk_pipe)
    fitted = xorq_pipe.fit(train_data, features=tuple(all_features), target=target)
    preds = fitted.predict(test_data, name="pred")

    make_metric = deferred_sklearn_metric(target=target, pred="pred")
    metrics = preds.agg(
        mae=make_metric(metric=mean_absolute_error),
        mse=make_metric(metric=mean_squared_error),
    )

    plot_expr = deferred_matplotlib_plot(preds, _build_xorq_figure)

    return plot_expr, metrics


# =========================================================================
# Run and plot side by side
# =========================================================================

def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()

    print("=== SKLEARN WAY ===")
    sk = sklearn_way(df)

    print("\n=== XORQ WAY ===")
    plot_expr, metrics = xorq_way(df)

    # Execute deferred expressions
    metrics_df = metrics.execute()
    xo_mae = metrics_df["mae"].iloc[0]
    xo_rmse = np.sqrt(metrics_df["mse"].iloc[0])
    print(f"  xorq:   MAE = {xo_mae:.2f}, RMSE = {xo_rmse:.2f}")

    xo_png = plot_expr.execute()

    # Build sklearn subplot natively
    sk_fig, sk_ax = plt.subplots(figsize=(8, 5))
    hours = range(N_PLOT)
    sk_ax.plot(
        hours, sk["y_test"][-N_PLOT:], color="black", linewidth=1.2, label="actual"
    )
    sk_ax.plot(
        hours,
        sk["y_pred"][-N_PLOT:],
        color="tab:blue",
        linewidth=1,
        linestyle="--",
        label=f"predicted (MAE={sk['mae']:.2f})",
    )
    sk_ax.set_title("sklearn - HGBR with lagged features")
    sk_ax.set_xlabel("Hours")
    sk_ax.set_ylabel("Bike count")
    sk_ax.legend(fontsize=9)
    plt.tight_layout()

    # Composite: sklearn (left) | xorq deferred plot (right)
    xo_img = load_plot_bytes(xo_png)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn")
    axes[0].axis("off")

    axes[1].imshow(xo_img)
    axes[1].set_title("xorq")
    axes[1].axis("off")

    plt.suptitle(
        "Lagged Features Forecasting: sklearn vs xorq (last 96h)", fontsize=14
    )
    plt.tight_layout()
    out = "imgs/time_series_lagged_features.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
