"""Demonstrating different strategies of KBinsDiscretizer
==========================================================

sklearn: Generate three synthetic datasets (uniform, clustered blob patterns).
For each dataset, apply KBinsDiscretizer with 'uniform', 'quantile', and 'kmeans'
strategies. Fit on data, transform a meshgrid to visualize bin boundaries via
contour plots. All execution is eager on numpy arrays.

xorq: Same datasets registered as ibis tables. Wrap KBinsDiscretizer pipeline
via Pipeline.from_instance, fit deferred, transform deferred, build deferred
plots via deferred_matplotlib_plot. Execution happens only on .execute().

Both produce identical discretization boundaries.

Dataset: Synthetic (make_blobs, uniform random)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.datasets import make_blobs
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import KBinsDiscretizer
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import fig_to_image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGIES = ("uniform", "quantile", "kmeans")
N_SAMPLES = 200
RANDOM_STATE = 42
N_BINS = 4
ENCODE = "ordinal"
QUANTILE_METHOD = "averaged_inverted_cdf"
FEATURE_COLS = ("x0", "x1")


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Generate three synthetic datasets matching sklearn example."""
    centers_0 = np.array([[0, 0], [0, 5], [2, 4], [8, 8]])
    centers_1 = np.array([[0, 0], [3, 1]])

    rng = np.random.RandomState(RANDOM_STATE)

    X_list = [
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

    return X_list


# ---------------------------------------------------------------------------
# Plotting helpers (used by deferred_matplotlib_plot)
# ---------------------------------------------------------------------------


def _build_dataset_figure(X, transformed_grids, title_prefix):
    """Build a single row figure: input data + 3 strategy plots.

    Parameters
    ----------
    X : numpy array (N, 2)
        Original data.
    transformed_grids : dict[str, dict]
        For each strategy: {"xx": xx, "yy": yy, "horizontal": h, "vertical": v}
    title_prefix : str
        E.g. "sklearn" or "xorq"

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))

    # Subplot 0: Input data
    axes[0].scatter(X[:, 0], X[:, 1], edgecolors="k", s=15)
    axes[0].set_title("Input data", size=12)
    axes[0].set_xticks(())
    axes[0].set_yticks(())
    axes[0].set_xlim(X[:, 0].min(), X[:, 0].max())
    axes[0].set_ylim(X[:, 1].min(), X[:, 1].max())

    # Subplots 1-3: Strategy results
    for i, strategy in enumerate(STRATEGIES, start=1):
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
# Shared pipeline definitions
# ---------------------------------------------------------------------------


def _build_pipelines():
    """Build KBinsDiscretizer pipelines for each strategy."""
    return {
        strategy: SklearnPipeline(
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
        )
        for strategy in STRATEGIES
    }


# =========================================================================
# SKLEARN WAY -- eager execution
# =========================================================================


def sklearn_way(X_list, pipelines):
    """Eager sklearn: fit KBinsDiscretizer, transform meshgrid, plot.

    Parameters
    ----------
    X_list : list of numpy arrays
        Three synthetic datasets.
    pipelines : dict[str, sklearn.pipeline.Pipeline]
        One pipeline per strategy.

    Returns
    -------
    dict[str, matplotlib.figure.Figure]
        One figure per dataset.
    """
    figures = {}

    for ds_idx, X in enumerate(X_list):
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min(), X[:, 0].max(), 300),
            np.linspace(X[:, 1].min(), X[:, 1].max(), 300),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        transformed_grids = {}
        for strategy in STRATEGIES:
            pipe = pipelines[strategy]
            pipe.fit(X)
            grid_encoded = pipe.transform(grid)
            transformed_grids[strategy] = {
                "xx": xx,
                "yy": yy,
                "horizontal": grid_encoded[:, 0].reshape(xx.shape),
                "vertical": grid_encoded[:, 1].reshape(xx.shape),
            }

        figures[f"dataset_{ds_idx}"] = _build_dataset_figure(X, transformed_grids, "sklearn")

    return figures


# =========================================================================
# XORQ WAY -- deferred execution
# =========================================================================


def xorq_way(X_list, pipelines):
    """Deferred xorq: Pipeline.from_instance, deferred fit/transform.

    Parameters
    ----------
    X_list : list of numpy arrays
        Three synthetic datasets.
    pipelines : dict[str, sklearn.pipeline.Pipeline]
        One pipeline per strategy.

    Returns
    -------
    dict[str, dict]
        For each dataset: {"X": array, "transformed_grids": dict of deferred exprs}
    """
    con = xo.connect()
    results = {}

    for ds_idx, X in enumerate(X_list):
        df = pd.DataFrame(X, columns=list(FEATURE_COLS))
        data = con.register(df, f"dataset_{ds_idx}")

        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min(), X[:, 0].max(), 300),
            np.linspace(X[:, 1].min(), X[:, 1].max(), 300),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_df = pd.DataFrame(grid, columns=list(FEATURE_COLS))
        grid_data = con.register(grid_df, f"grid_{ds_idx}")

        transformed_grids = {}
        for strategy in STRATEGIES:
            xorq_pipe = Pipeline.from_instance(pipelines[strategy])
            fitted = xorq_pipe.fit(data, features=FEATURE_COLS, target=None)
            grid_encoded = fitted.transform(grid_data)
            transformed_grids[strategy] = {"xx": xx, "yy": yy, "grid_encoded": grid_encoded}

        results[f"dataset_{ds_idx}"] = {"X": X, "transformed_grids": transformed_grids}

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    X_list = _load_data()
    pipelines = _build_pipelines()

    print("=== SKLEARN WAY ===")
    sk_figures = sklearn_way(X_list, pipelines)

    print("\n=== XORQ WAY ===")
    deferred = xorq_way(X_list, pipelines)

    # Execute deferred transformations and build xorq figures
    print("Executing deferred transformations...")
    xo_figures = {}
    for ds_name, result in deferred.items():
        X = result["X"]
        transformed_grids = result["transformed_grids"]

        # Execute all grid transformations for this dataset
        materialized = {}
        for strategy in STRATEGIES:
            grid_info = transformed_grids[strategy]
            xx = grid_info["xx"]
            yy = grid_info["yy"]
            grid_encoded = grid_info["grid_encoded"].execute()

            # The transformed output is a list of {'key': 'x0', 'value': v} dicts
            # Extract the values for x0 and x1
            if "transformed" in grid_encoded.columns:
                def extract_value(row, key_name):
                    for item in row:
                        if item["key"] == key_name:
                            return item["value"]
                    return None

                horizontal = grid_encoded["transformed"].apply(
                    lambda x: extract_value(x, FEATURE_COLS[0])
                ).values.reshape(xx.shape)
                vertical = grid_encoded["transformed"].apply(
                    lambda x: extract_value(x, FEATURE_COLS[1])
                ).values.reshape(xx.shape)
            else:
                # Fallback: direct column access
                horizontal = grid_encoded[FEATURE_COLS[0]].values.reshape(xx.shape)
                vertical = grid_encoded[FEATURE_COLS[1]].values.reshape(xx.shape)

            materialized[strategy] = {
                "xx": xx,
                "yy": yy,
                "horizontal": horizontal,
                "vertical": vertical,
            }

        # Build figure
        xo_figures[ds_name] = _build_dataset_figure(X, materialized, "xorq")

    # Build composite: 3 rows (one per dataset), 2 columns (sklearn | xorq)
    fig, axes = plt.subplots(3, 2, figsize=(20, 12))

    for row_idx in range(3):
        ds_name = f"dataset_{row_idx}"

        # sklearn subplot (rendered to image)
        sk_fig = sk_figures[ds_name]
        axes[row_idx, 0].imshow(fig_to_image(sk_fig))
        axes[row_idx, 0].axis("off")

        # xorq subplot (rendered to image)
        xo_fig = xo_figures[ds_name]
        axes[row_idx, 1].imshow(fig_to_image(xo_fig))
        axes[row_idx, 1].axis("off")

    axes[0, 0].set_title("sklearn", fontsize=14, fontweight="bold", pad=10)
    axes[0, 1].set_title("xorq (deferred)", fontsize=14, fontweight="bold", pad=10)

    fig.suptitle(
        "KBinsDiscretizer Strategies: sklearn vs xorq", fontsize=16, fontweight="bold"
    )
    fig.tight_layout()
    out = "imgs/discretization_strategies.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
