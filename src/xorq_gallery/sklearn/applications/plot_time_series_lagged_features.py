"""Lagged features for time series forecasting
================================================

sklearn: Engineer lagged features with pandas shift, fit
HistGradientBoostingRegressor on numpy arrays, evaluate MAE/RMSE.
Split with train_test_split(shuffle=False) to preserve time order.

xorq: Same lagged features computed in load_data() via pandas, same
model wrapped in Pipeline.from_instance, fit/predict deferred, metrics
via deferred_sklearn_metric.

Both splits preserve temporal order and produce identical train/test
rows independently.

Dataset: Bike Sharing Demand (OpenML)
"""

from functools import cache, partial

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
)
from xorq_gallery.utils import (
    fig_to_image,
    save_fig,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROW_IDX = "row_idx"
PRED_COL = "pred"
TARGET_COL = "count"
CALENDAR_FEATURES = ("hour", "weekday", "month")
WEATHER_FEATURES = ("temp", "humidity", "windspeed")
MODEL_PARAMS = dict(max_iter=200, max_depth=8, learning_rate=0.1, random_state=42)

LAGGED_FEATURES_NS = (
    ("lagged_count_1h", 1),
    ("lagged_count_2h", 2),
    ("lagged_count_3h", 3),
    ("lagged_count_1d", 24),
    ("lagged_count_7d", 24 * 7),
)

LAGGED_FEATURES = tuple(name for name, _ in LAGGED_FEATURES_NS)
FEATURE_COLS = LAGGED_FEATURES + CALENDAR_FEATURES + WEATHER_FEATURES

N_PLOT = 96


# ---------------------------------------------------------------------------
# Data loading (shared) — lags computed in pandas
# ---------------------------------------------------------------------------


@cache
def load_data():
    """Load Bike Sharing Demand and engineer lagged features via pandas shift."""
    bike_sharing = fetch_openml(
        "Bike_Sharing_Demand", version=2, as_frame=True, parser="pandas"
    )
    df = bike_sharing.frame
    df = (
        df.astype({name: float for name in ("count", "temp", "humidity", "windspeed")})
        .astype({name: int for name in ("hour", "weekday", "month")})
        .assign(**{ROW_IDX: range(len(df))})
        .sort_values(ROW_IDX)
    )
    # Normalise target
    df[TARGET_COL] = df[TARGET_COL] / df[TARGET_COL].max()

    # Compute lags via pandas shift (data already sorted by ROW_IDX)
    df = df.assign(
        **{
            lagged_feature: df[TARGET_COL].shift(n)
            for lagged_feature, n in LAGGED_FEATURES_NS
        }
    ).dropna(subset=list(LAGGED_FEATURES))

    return df[list(FEATURE_COLS) + [TARGET_COL]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def compare_results(comparator):
    print("\n=== Comparing Results ===")
    for name in comparator.sklearn_results:
        sk = comparator.sklearn_results[name]
        xo = comparator.xorq_results[name]
        sk_rmse = np.sqrt(sk["metrics"]["mse"])
        xo_rmse = np.sqrt(xo["metrics"]["mse"])
        print(f"  sklearn {name}: MAE={sk['metrics']['mae']:.4f}  RMSE={sk_rmse:.4f}")
        print(f"  xorq    {name}: MAE={xo['metrics']['mae']:.4f}  RMSE={xo_rmse:.4f}")


def _build_forecast_figure(actual, preds, mae, label):
    fig, ax = plt.subplots(figsize=(8, 5))
    hours = range(N_PLOT)
    ax.plot(hours, actual[-N_PLOT:], color="black", linewidth=1.2, label="actual")
    ax.plot(
        hours,
        preds[-N_PLOT:],
        color="tab:blue",
        linewidth=1,
        linestyle="--",
        label=f"predicted (MAE={mae:.4f})",
    )
    ax.set_title(f"{label} - HGBR with lagged features")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Bike count (normalised)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_results(comparator):
    _train, test = comparator.get_split_data()
    y_actual = test[TARGET_COL].values

    sk_result = comparator.sklearn_results[HGBR]
    xo_result = comparator.xorq_results[HGBR]

    sk_fig = _build_forecast_figure(
        y_actual, sk_result["preds"], sk_result["metrics"]["mae"], "sklearn"
    )
    xo_preds_df = xo_result["preds"]
    xo_fig = _build_forecast_figure(
        xo_preds_df[TARGET_COL].values,
        xo_preds_df[PRED_COL].values,
        xo_result["metrics"]["mae"],
        "xorq",
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn")
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq")
    axes[1].axis("off")

    fig.suptitle("Lagged Features Forecasting: sklearn vs xorq (last 96h)", fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = (HGBR,) = ("HGBR",)
names_pipelines = (
    (HGBR, SklearnPipeline([("hgbr", HistGradientBoostingRegressor(**MODEL_PARAMS))])),
)
metrics_names_funcs = (
    ("mae", mean_absolute_error),
    ("mse", mean_squared_error),
)

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=TARGET_COL,
    pred=PRED_COL,
    metrics_names_funcs=metrics_names_funcs,
    load_data=load_data,
    split_data=partial(train_test_split, test_size=0.333, shuffle=False),
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)
# expose the exprs to invoke `xorq build plot_time_series_lagged_features.py --expr $expr_name`
(xorq_hgbr_preds,) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    comparator.result_comparison
    save_fig("imgs/time_series_lagged_features.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
