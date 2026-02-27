"""L1-based models for Sparse Signals
====================================

sklearn: Generate synthetic sparse signals with correlated features, fit Lasso,
ARDRegression, and ElasticNet models eagerly on numpy arrays, evaluate R^2 score
and compare estimated coefficients with ground-truth. Split with
train_test_split(shuffle=False) to preserve temporal order.

xorq: Same models wrapped in Pipeline.from_instance. Data is an ibis expression,
split via deferred_sequential_split, fit/predict deferred, metrics via
deferred_sklearn_metric, coefficients extracted via deferred UDAFs.

Both produce identical R^2 scores and coefficient patterns.

Dataset: Synthetic sparse sinusoidal signals with Gaussian noise
"""

from __future__ import annotations

from functools import partial
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toolz
from matplotlib.colors import SymLogNorm
from sklearn.base import clone
from sklearn.linear_model import ARDRegression, ElasticNet, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.sklearn.sklearn_lib import SklearnXorqComparator
from xorq_gallery.utils import (
    fig_to_image,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SAMPLES = 50
N_FEATURES = 100
N_INFORMATIVE = 10
RANDOM_STATE = 0

# Fixed hyperparameters (in practice, use LassoCV / ElasticNetCV)
LASSO_ALPHA = 0.14
ELASTICNET_ALPHA = 0.08
ELASTICNET_L1_RATIO = 0.5

# Column names
FEATURE_COLS = tuple(f"feature_{i}" for i in range(N_FEATURES))
TARGET_COL = "target"
ROW_IDX = "row_idx"
PRED_COL = "pred"


# ---------------------------------------------------------------------------
# Data generation (shared)
# ---------------------------------------------------------------------------


def _generate_sparse_signal():
    """Generate synthetic sparse signal with correlated features.

    Returns ground-truth coefficients and (X, y) data following the sklearn
    example exactly.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    time_step = np.linspace(-2, 2, N_SAMPLES)
    freqs = 2 * np.pi * np.sort(rng.rand(N_FEATURES)) / 0.01

    # Generate X as sinusoids
    X = np.zeros((N_SAMPLES, N_FEATURES))
    for i in range(N_FEATURES):
        X[:, i] = np.sin(freqs[i] * time_step)

    # True coefficients: sparse, alternating signs, exponential decay
    idx = np.arange(N_FEATURES)
    true_coef = (-1) ** idx * np.exp(-idx / 10)
    true_coef[N_INFORMATIVE:] = 0

    # Generate y
    y = np.dot(X, true_coef)

    # Add random phase and Gaussian noise
    for i in range(N_FEATURES):
        X[:, i] = np.sin(freqs[i] * time_step + 2 * (rng.random_sample() - 0.5))
        X[:, i] += 0.2 * rng.normal(0, 1, N_SAMPLES)

    y += 0.2 * rng.normal(0, 1, N_SAMPLES)

    return true_coef, X, y, time_step


def load_data():
    """Load data as pandas DataFrame with row_idx for temporal ordering."""
    true_coef, X, y, time_step = _generate_sparse_signal()

    df = pd.DataFrame(X, columns=FEATURE_COLS).assign(
        **{
            TARGET_COL: y,
            "time_step": time_step,
            ROW_IDX: range(len(X)),
        }
    )
    df.attrs["true_coef"] = true_coef

    return df


@toolz.curry
def split_data(f, features, target, df):
    X, y = (df[list(features)], df[target])
    X_train, X_test, y_train, y_test = f(X, y)
    train_data, test_data = (
        pd.DataFrame(X, columns=features).assign(**{target: y})
        for X, y in (
            (X_train, y_train),
            (X_test, y_test),
        )
    )
    return train_data, test_data


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _build_coefficient_heatmap(coef_matrix, row_labels, r2_scores, title_prefix):
    """Build coefficient comparison heatmap using matplotlib.

    Parameters
    ----------
    coef_matrix : np.ndarray
        2D array of coefficients (models x features)
    row_labels : list[str]
        Labels for each row (model names)
    r2_scores : dict
        Dictionary with keys "Lasso", "ARD", "ElasticNet" -> float R^2 scores
    title_prefix : str
        Prefix for the title (e.g., "sklearn" or "xorq")

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use imshow with SymLogNorm
    im = ax.imshow(
        coef_matrix,
        aspect="auto",
        cmap="seismic_r",
        norm=SymLogNorm(linthresh=10e-4, vmin=-1, vmax=1),
    )

    # Set ticks and labels
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_ylabel("linear model")
    ax.set_xlabel("coefficients")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("coefficients' values")

    ax.set_title(
        f"{title_prefix} - Models' coefficients\n"
        f"Lasso $R^2$: {r2_scores['Lasso']:.3f}, "
        f"ARD $R^2$: {r2_scores['ARD']:.3f}, "
        f"ElasticNet $R^2$: {r2_scores['ElasticNet']:.3f}"
    )
    fig.tight_layout()
    return fig


def make_deferred_xorq_result(
    pipeline, train_data, test_data, features, target, metrics_names_funcs, pred
):
    xorq_fitted = Pipeline.from_instance(pipeline).fit(
        train_data, features=features, target=target
    )
    preds = xorq_fitted.predict(test_data, name=pred)
    metrics = {
        name: preds.agg(
            **{
                name: deferred_sklearn_metric(
                    target=target, pred=pred, metric=metric_fn
                )
            }
        )
        for name, metric_fn in metrics_names_funcs
    }
    deferred_xorq_result = {
        "xorq_fitted": xorq_fitted,
        "preds": preds,
        "metrics": metrics,
    }
    return deferred_xorq_result


def make_xorq_result(deferred_xorq_result):
    xorq_fitted, preds, metrics = (
        deferred_xorq_result[name]
        for name in (
            "xorq_fitted",
            "preds",
            "metrics",
        )
    )
    result = {
        "fitted_model": xorq_fitted.fitted_steps[-1].model,
        "preds": preds.execute(),
        "metrics": {name: expr.as_scalar().execute() for name, expr in metrics.items()},
        "other": {
            "coef": xorq_fitted.fitted_steps[-1].model.coef_,
        },
    }
    return result


def make_sklearn_result(
    pipeline, train_data, test_data, features, target, metrics_names_funcs
):
    ((X_train, y_train), (X_test, y_test)) = (
        (df[list(features)], df[target]) for df in (train_data, test_data)
    )
    t0 = time()
    fitted = clone(pipeline).fit(X_train, y_train)
    fit_time = time() - t0
    preds = fitted.predict(X_test)
    metrics = {
        metric_name: metric_func(y_test, preds)
        for metric_name, metric_func in metrics_names_funcs
    }
    result = {
        "fitted_model": fitted.steps[-1],
        "preds": preds,
        "metrics": metrics,
        "other": {
            "coef": fitted[-1].coef_,
            "fit_time": fit_time,
        },
    }
    return result


def compare_result(name, sklearn_result, xorq_result):
    # ---- Compare result ----
    # Note: Results may differ slightly due to subtle differences in how
    # TimeSeriesSplit and train_test_split handle the splits. Both methods
    # preserve temporal order and use roughly the same train/test proportions,
    # but TimeSeriesSplit uses fold-based splitting while train_test_split
    # uses a simple cutoff.
    sk_r2, xo_r2 = (dct["metrics"]["r2"] for dct in (sklearn_result, xorq_result))
    r2_diff = abs(sk_r2 - xo_r2)
    print(f"{name} R^2 - sklearn: {sk_r2:.3f}, xorq: {xo_r2:.3f}, diff: {r2_diff:.4f}")

    sk_coef, xo_coef = (dct["other"]["coef"] for dct in (sklearn_result, xorq_result))
    coef_diff = np.max(np.abs(sk_coef - xo_coef))
    print(f"{name} max coef difference: {coef_diff:.6f}")


def compare_results(comparator):
    assert sorted(sklearn_results := comparator.sklearn_results) == sorted(
        xorq_results := comparator.xorq_results
    )
    print("\n=== Comparing Results ===")
    for name, sklearn_result in sklearn_results.items():
        xorq_result = xorq_results[name]
        compare_result(name, sklearn_result, xorq_result)


def plot_results(comparator):
    names = tuple(name for name, _ in comparator.names_pipelines)

    # Build coefficient matrices
    true_coef = comparator.df.attrs["true_coef"]
    sk_coefs, xo_coefs = (
        tuple(result["other"]["coef"] for result in (results[name] for name in names))
        for results in (comparator.sklearn_results, comparator.xorq_results)
    )
    row_labels = ["True coefficients", *names]
    sk_coef_matrix, xo_coef_matrix = (
        np.vstack(
            [
                true_coef,
                *coefs,
            ]
        )
        for coefs in (sk_coefs, xo_coefs)
    )

    # Build r2 scores
    (sk_r2_scores, xo_r2_scores) = (
        {name: results[name]["metrics"]["r2"] for name in names}
        for results in (comparator.sklearn_results, comparator.xorq_results)
    )

    # Build sklearn heatmap
    sk_fig = _build_coefficient_heatmap(
        sk_coef_matrix, row_labels, sk_r2_scores, "sklearn"
    )

    # Build xorq heatmap
    xo_fig = _build_coefficient_heatmap(
        xo_coef_matrix, row_labels, xo_r2_scores, "xorq"
    )

    # Composite: sklearn (left) | xorq (right)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].axis("off")

    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].axis("off")

    fig.suptitle("L1-based models for Sparse Signals: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    return fig


methods = (LASSO, ARD, ELASTICNET) = ("Lasso", "ARD", "ElasticNet")
names_pipelines = (
    (LASSO, SklearnPipeline([("lasso", Lasso(alpha=LASSO_ALPHA))])),
    (ARD, SklearnPipeline([("ard", ARDRegression())])),
    (
        ELASTICNET,
        SklearnPipeline(
            [
                (
                    "elasticnet",
                    ElasticNet(alpha=ELASTICNET_ALPHA, l1_ratio=ELASTICNET_L1_RATIO),
                )
            ]
        ),
    ),
)
metrics_names_funcs = (("r2", r2_score),)


comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=TARGET_COL,
    pred=PRED_COL,
    metrics_names_funcs=metrics_names_funcs,
    load_data=load_data,
    split_data=partial(train_test_split, test_size=0.3333, shuffle=False),
    make_sklearn_result=make_sklearn_result,
    make_deferred_xorq_result=make_deferred_xorq_result,
    make_xorq_result=make_xorq_result,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)
# expose the exprs in the script to invoke `xorq build plot_lasso_and_elasticnet.py --expr $expr_name`
(xorq_lasso_preds, xorq_ard_preds, xorq_elastic_net_preds) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    print(
        "\nBoth approaches produce L1-regularized sparse models with similar sparsity patterns."
    )
    comparator.result_comparison
    comparator.save_comparison_plot("imgs/lasso_and_elasticnet.png")


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
