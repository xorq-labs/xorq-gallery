"""Demonstrating different strategies of KBinsDiscretizer
==========================================================

sklearn: Generate three synthetic datasets (uniform, clustered blob patterns).
For each dataset, apply KBinsDiscretizer with 'uniform', 'quantile', and 'kmeans'
strategies. Fit on data, transform a meshgrid to visualize bin boundaries via
contour plots. All execution is eager on numpy arrays.

xorq: Same KBinsDiscretizer pipelines wrapped in Pipeline.from_instance. Each
transformation is computed lazily as a deferred expression and materialized
when executed. Meshgrid transformations use xorq_fitted.transform directly.

Both produce identical discretization boundaries.

Dataset: Synthetic (make_blobs, uniform random)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/preprocessing/plot_discretization_strategies.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import KBinsDiscretizer
from xorq.api import SessionConfig
from xorq.config import options

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    make_deferred_xorq_fit_transform_result,
    make_sklearn_fit_transform_result,
    make_xorq_fit_transform_result,
    split_data_nop,
)
from xorq_gallery.utils import fig_to_image, save_fig


# Force single-threaded DataFusion to preserve scan order for UDAF
options.backend = xo.connect(session_config=SessionConfig().with_target_partitions(1))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SAMPLES = 200
RANDOM_STATE = 42
N_BINS = 4
ENCODE = "ordinal"
QUANTILE_METHOD = "averaged_inverted_cdf"
FEATURE_COLS = ("x0", "x1")
TARGET_COL = "dataset_id"
PRED_COL = "pred"  # unused, required by comparator


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_raw_datasets():
    """Generate three synthetic datasets matching sklearn example."""
    centers_0 = np.array([[0, 0], [0, 5], [2, 4], [8, 8]])
    centers_1 = np.array([[0, 0], [3, 1]])
    rng = np.random.RandomState(RANDOM_STATE)
    return [
        rng.uniform(-3, 3, size=(N_SAMPLES, 2)),
        make_blobs(
            n_samples=[
                N_SAMPLES // 10,
                N_SAMPLES * 4 // 10,
                N_SAMPLES // 10,
                N_SAMPLES * 4 // 10,
            ],
            cluster_std=0.5,
            centers=centers_0,
            random_state=RANDOM_STATE,
        )[0],
        make_blobs(
            n_samples=[N_SAMPLES // 5, N_SAMPLES * 4 // 5],
            cluster_std=0.5,
            centers=centers_1,
            random_state=RANDOM_STATE,
        )[0],
    ]


_raw_datasets = _load_raw_datasets()


def _make_load_data(ds_idx):
    def load_data():
        X = _raw_datasets[ds_idx]
        df = pd.DataFrame(X, columns=list(FEATURE_COLS))
        df[TARGET_COL] = ds_idx
        return df

    return load_data


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _unnest_transformed(df):
    """Extract FEATURE_COLS from nested key-value 'transformed' column.

    KBinsDiscretizer xorq output wraps results in a list of {'key', 'value'}
    dicts under a 'transformed' column rather than flat feature columns.
    """
    result = {}
    for col in FEATURE_COLS:
        result[col] = df["transformed"].apply(
            lambda items, c=col: next(
                item["value"] for item in items if item["key"] == c
            )
        )
    return pd.DataFrame(result)


def _make_grid(comparator):
    """Compute 300×300 meshgrid from comparator.df bounds."""
    df = comparator.df
    X = df[list(FEATURE_COLS)].values
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 0].max(), 300),
        np.linspace(X[:, 1].min(), X[:, 1].max(), 300),
    )
    grid_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=list(FEATURE_COLS))
    return xx, yy, grid_df


def _build_dataset_figure(X, transformed_grids, title_prefix):
    """Build a 4-panel row: input data + 3 strategy contour plots."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))

    axes[0].scatter(X[:, 0], X[:, 1], edgecolors="k", s=15)
    axes[0].set_title("Input data", size=12)
    axes[0].set_xticks(())
    axes[0].set_yticks(())
    axes[0].set_xlim(X[:, 0].min(), X[:, 0].max())
    axes[0].set_ylim(X[:, 1].min(), X[:, 1].max())

    for i, strategy in enumerate(methods, start=1):
        grid_data = transformed_grids[strategy]
        xx = grid_data["xx"]
        yy = grid_data["yy"]
        horizontal = grid_data["horizontal"]
        vertical = grid_data["vertical"]

        axes[i].contourf(xx, yy, horizontal, alpha=0.5, cmap="viridis")
        axes[i].contourf(xx, yy, vertical, alpha=0.5, cmap="plasma")
        axes[i].scatter(X[:, 0], X[:, 1], edgecolors="k", s=15)
        axes[i].set_title(f"strategy='{strategy}'", size=12)
        axes[i].set_xticks(())
        axes[i].set_yticks(())
        axes[i].set_xlim(xx.min(), xx.max())
        axes[i].set_ylim(yy.min(), yy.max())

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def compare_results(comparator):
    print("\n=== Comparing Results ===")
    for name in comparator.sklearn_results:
        sk_X = comparator.sklearn_results[name]["transformed"]
        xo_raw = comparator.xorq_results[name]["transformed"]
        xo_X = _unnest_transformed(xo_raw).values
        np.testing.assert_allclose(sk_X, xo_X, rtol=1e-5, atol=1e-5)
        print(f"  {name}: sklearn vs xorq match (shape {sk_X.shape})")


def plot_results(comparator):
    df = comparator.df
    X = df[list(FEATURE_COLS)].values
    xx, yy, grid_df = _make_grid(comparator)

    sk_grids = {
        name: {
            "xx": xx,
            "yy": yy,
            "horizontal": comparator.sklearn_results[name]["fitted"]
            .transform(grid_df[list(FEATURE_COLS)])[:, 0]
            .reshape(xx.shape),
            "vertical": comparator.sklearn_results[name]["fitted"]
            .transform(grid_df[list(FEATURE_COLS)])[:, 1]
            .reshape(xx.shape),
        }
        for name, _ in comparator.names_pipelines
    }

    xo_grids = {}
    for name, _ in comparator.names_pipelines:
        xorq_fitted = comparator.deferred_xorq_results[name]["xorq_fitted"]
        grid_encoded = _unnest_transformed(
            xorq_fitted.transform(xo.memtable(grid_df)).execute()
        )
        xo_grids[name] = {
            "xx": xx,
            "yy": yy,
            "horizontal": grid_encoded[FEATURE_COLS[0]].values.reshape(xx.shape),
            "vertical": grid_encoded[FEATURE_COLS[1]].values.reshape(xx.shape),
        }

    sk_fig = _build_dataset_figure(X, sk_grids, "sklearn")
    xo_fig = _build_dataset_figure(X, xo_grids, "xorq")

    fig, axes = plt.subplots(1, 2, figsize=(28, 3))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].axis("off")
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold", pad=10)
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].axis("off")
    axes[1].set_title("xorq", fontsize=14, fontweight="bold", pad=10)
    plt.close(sk_fig)
    plt.close(xo_fig)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = (UNIFORM, QUANTILE, KMEANS) = ("uniform", "quantile", "kmeans")
names_pipelines = tuple(
    (
        strategy,
        SklearnPipeline(
            [
                (
                    "discretizer",
                    KBinsDiscretizer(
                        n_bins=N_BINS,
                        encode=ENCODE,
                        quantile_method=QUANTILE_METHOD,
                        strategy=strategy,
                    ),
                )
            ]
        ),
    )
    for strategy in methods
)
metrics_names_funcs = ()

_comparator_kwargs = dict(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=TARGET_COL,
    pred=PRED_COL,
    metrics_names_funcs=metrics_names_funcs,
    split_data=split_data_nop,
    make_sklearn_result=make_sklearn_fit_transform_result,
    make_deferred_xorq_result=make_deferred_xorq_fit_transform_result,
    make_xorq_result=make_xorq_fit_transform_result,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)

comparator_ds0 = SklearnXorqComparator(
    load_data=_make_load_data(0), **_comparator_kwargs
)
comparator_ds1 = SklearnXorqComparator(
    load_data=_make_load_data(1), **_comparator_kwargs
)
comparator_ds2 = SklearnXorqComparator(
    load_data=_make_load_data(2), **_comparator_kwargs
)

# expose the exprs to invoke `xorq build plot_discretization_strategies.py --expr $expr_name`
(
    xorq_ds0_uniform_transformed,
    xorq_ds0_quantile_transformed,
    xorq_ds0_kmeans_transformed,
) = (comparator_ds0.deferred_xorq_results[name]["transformed"] for name in methods)
(
    xorq_ds1_uniform_transformed,
    xorq_ds1_quantile_transformed,
    xorq_ds1_kmeans_transformed,
) = (comparator_ds1.deferred_xorq_results[name]["transformed"] for name in methods)
(
    xorq_ds2_uniform_transformed,
    xorq_ds2_quantile_transformed,
    xorq_ds2_kmeans_transformed,
) = (comparator_ds2.deferred_xorq_results[name]["transformed"] for name in methods)


def main():
    for comparator in (comparator_ds0, comparator_ds1, comparator_ds2):
        comparator.result_comparison

    ds_figs = [
        comparator.plot_results()
        for comparator in (comparator_ds0, comparator_ds1, comparator_ds2)
    ]
    fig, axes = plt.subplots(3, 1, figsize=(28, 9))
    for row, ds_fig in enumerate(ds_figs):
        axes[row].imshow(fig_to_image(ds_fig))
        axes[row].axis("off")
        plt.close(ds_fig)
    fig.suptitle(
        "KBinsDiscretizer Strategies: sklearn vs xorq", fontsize=16, fontweight="bold"
    )
    fig.tight_layout()
    save_fig("imgs/discretization_strategies.png", fig)


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
