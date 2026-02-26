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

from functools import cache

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
    save_fig,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROW_IDX = "row_idx"
pred_col = "pred"
target = "count"
calendar_features = ("hour", "weekday", "month")
weather_features = ("temp", "humidity", "windspeed")
model_params = dict(max_iter=200, max_depth=8, learning_rate=0.1, random_state=42)
lagged_features_ns = (
    ("lagged_count_1h", 1),
    ("lagged_count_2h", 2),
    ("lagged_count_3h", 3),
    ("lagged_count_1d", 24),
    ("lagged_count_7d", 24 * 7),
)

lagged_features = tuple(name for name, _ in lagged_features_ns)
all_features = lagged_features + calendar_features + weather_features

hgbr_pipeline = SklearnPipeline(
    [("hgbr", HistGradientBoostingRegressor(**model_params))]
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
        df.astype({name: float for name in ("count", "temp", "humidity", "windspeed")})
        .astype({name: int for name in ("hour", "weekday", "month")})
        .assign(**{ROW_IDX: range(len(df))})
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

N_PLOT = 96


def _build_sklearn_figure(sk):
    """Plot sklearn predictions vs actuals for the last N_PLOT hours."""
    fig, ax = plt.subplots(figsize=(8, 5))
    hours = range(N_PLOT)
    ax.plot(hours, sk["y_test"][-N_PLOT:], color="black", linewidth=1.2, label="actual")
    ax.plot(
        hours,
        sk["y_pred"][-N_PLOT:],
        color="tab:blue",
        linewidth=1,
        linestyle="--",
        label=f"predicted (MAE={sk['mae']:.2f})",
    )
    ax.set_title("sklearn - HGBR with lagged features")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Bike count")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def _build_xorq_figure(df):
    """Plot xorq predictions vs actuals for the last N_PLOT hours."""
    actual = df[target].values
    predicted = df[pred_col].values
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
    fig.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- lagged features via pandas, eager split
# =========================================================================


def compute_with_sklearn(df):
    """Eager sklearn: pandas shift for lags, shuffle=False split."""
    ldf = (
        df[list(("count", ROW_IDX) + calendar_features + weather_features)]
        .assign(
            **{
                lagged_feature: lambda d, n=n: d["count"].shift(n)
                for lagged_feature, n in lagged_features_ns
            }
        )
        .dropna()
    )

    feature_cols = [c for c in ldf.columns if c.startswith("lagged_")] + list(
        calendar_features + weather_features
    )

    X = ldf[feature_cols]
    y = ldf[target]

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
# XORQ WAY -- lagged features via ibis, deferred split
# =========================================================================


def compute_with_xorq(xorq_metrics, xorq_plot):
    """Execute deferred xorq metrics and plot expressions.

    Returns
    -------
    tuple
        (xo_mae, xo_rmse, xo_png)
    """
    metrics_df = xorq_metrics.execute()
    xo_mae = metrics_df["mae"].iloc[0]
    xo_rmse = np.sqrt(metrics_df["mse"].iloc[0])
    print(f"  xorq:   MAE = {xo_mae:.2f}, RMSE = {xo_rmse:.2f}")
    xo_png = xorq_plot.execute()
    return xo_mae, xo_rmse, xo_png


con = xo.connect()
data = con.register(_load_data(), "bike_sharing")

data_with_lags = data.mutate(
    **{
        lagged_feature: data[target].lag(n).over(order_by=ROW_IDX)
        for (lagged_feature, n) in lagged_features_ns
    }
).drop_null(subset=lagged_features)

train_data, test_data = deferred_sequential_split(
    data_with_lags, features=all_features, target=target, order_by=ROW_IDX
)

fitted = Pipeline.from_instance(hgbr_pipeline).fit(
    train_data, features=all_features, target=target
)
xorq_preds = fitted.predict(test_data, name=pred_col)
xorq_metrics = xorq_preds.agg(
    mae=deferred_sklearn_metric(
        target=target, pred=pred_col, metric=mean_absolute_error
    ),
    mse=deferred_sklearn_metric(
        target=target, pred=pred_col, metric=mean_squared_error
    ),
)
xorq_plot = deferred_matplotlib_plot(xorq_preds, _build_xorq_figure)


# =========================================================================
# Run and plot side by side
# =========================================================================


def save_comparison_plot(sk, xo_png):
    sk_fig = _build_sklearn_figure(sk)
    xo_img = load_plot_bytes(xo_png)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn")
    axes[0].axis("off")
    axes[1].imshow(xo_img)
    axes[1].set_title("xorq")
    axes[1].axis("off")

    fig.suptitle("Lagged Features Forecasting: sklearn vs xorq (last 96h)", fontsize=14)
    fig.tight_layout()
    save_fig("imgs/time_series_lagged_features.png", fig)


def main():
    df = _load_data()

    print("=== SKLEARN WAY ===")
    sk = compute_with_sklearn(df)

    print("\n=== XORQ WAY ===")
    xo_mae, xo_rmse, xo_png = compute_with_xorq(xorq_metrics, xorq_plot)

    save_comparison_plot(sk, xo_png)


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
