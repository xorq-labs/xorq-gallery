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

import os
from time import time
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toolz
import xorq.api as xo
from matplotlib.colors import SymLogNorm
from sklearn.base import clone
from sklearn.linear_model import ARDRegression, ElasticNet, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_sequential_split,
    fig_to_image,
    save_fig,
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


@cache
def _load_data():
    """Load data as pandas DataFrame with row_idx for temporal ordering."""
    true_coef, X, y, time_step = _generate_sparse_signal()

    df = pd.DataFrame(X, columns=FEATURE_COLS).assign(**{
        TARGET_COL: y,
        "time_step": time_step,
        ROW_IDX: range(len(X)),
    })
    df.attrs["true_coef"] = true_coef

    return df


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


def _deferred_fit_xorq_pipeline(sklearn_pipeline, train_data, test_data):
    fitted = Pipeline.from_instance(sklearn_pipeline).fit(train_data, features=FEATURE_COLS, target=TARGET_COL)
    preds = fitted.predict(test_data, name=PRED_COL)
    return {
        "preds": preds,
        "metrics": preds.agg(r2=deferred_sklearn_metric(target=TARGET_COL, pred=PRED_COL, metric=r2_score)),
        "fitted_pipe": fitted,
    }


def compute_with_xorq(name_to_xorq_exprs):
    """Deferred xorq: sequential split, fit three L1-based models deferred.

    Executes deferred expressions for predictions and metrics, plus fitted pipelines
    for coefficient extraction after execution.
    Nothing is executed until ``.execute()``.

    Returns
    -------
    dict
        Keys: "Lasso", "ARD", "ElasticNet"
        Values: dict with "preds", "metrics", "fitted_pipe"
    """
    xo_r2_scores = {
        name: exprs["metrics"].execute()["r2"].iloc[0]
        for name, exprs in name_to_xorq_exprs.items()
    }
    xo_coefs = {
        # FittedStep.model executes
        name: exprs["fitted_pipe"].fitted_steps[-1].model.coef_
        for name, exprs in name_to_xorq_exprs.items()
    }

    for name, r2 in xo_r2_scores.items():
        print(f"  xorq   {name}: R^2 = {r2:.3f}")

    return xo_r2_scores, xo_coefs


# =========================================================================
# SKLEARN WAY -- eager, train_test_split(shuffle=False)
# =========================================================================


def _fit_sklearn_pipeline(name, pipeline, X_train, X_test, y_train, y_test):
    t0 = time()
    fitted = clone(pipeline).fit(X_train, y_train)
    fit_time = time() - t0
    r2 = r2_score(y_test, fitted.predict(X_test))
    print(f"  sklearn {name}: R^2 = {r2:.3f}, fit time = {fit_time:.3f}s")
    return {"r2": r2, "coef": fitted[-1].coef_, "fit_time": fit_time}


def compute_with_sklearn(name_to_pipeline, df):
    """Eager sklearn: time-ordered split, fit three L1-based models, score.

    Returns
    -------
    dict
        Keys: "Lasso", "ARD", "ElasticNet"
        Values: dict with "r2", "coef", "fit_time"
    """
    X = df[list(FEATURE_COLS)].values
    y = df[TARGET_COL].values

    # shuffle=False preserves temporal order: first rows train, last rows test
    # Using test_size=0.3333 to match deferred_sequential_split behavior
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3333, shuffle=False
    )

    results = {
        name: _fit_sklearn_pipeline(name, pipeline, X_train, X_test, y_train, y_test)
        for name, pipeline in name_to_pipeline.items()
    }
    sk_r2_scores, sk_coefs, sk_fit_times = (
        toolz.valmap(toolz.curried.get(key), results)
        for key in ("r2", "coef", "fit_time")
    )
    return sk_r2_scores, sk_coefs, sk_fit_times


methods = (LASSO, ARD, ELASTICNET) = ("Lasso", "ARD", "ElasticNet")
name_to_pipeline = {
    LASSO: SklearnPipeline([("lasso", Lasso(alpha=LASSO_ALPHA))]),
    ARD: SklearnPipeline([("ard", ARDRegression())]),
    ELASTICNET: SklearnPipeline([("elasticnet", ElasticNet(alpha=ELASTICNET_ALPHA, l1_ratio=ELASTICNET_L1_RATIO))]),
}


# =========================================================================
# XORQ WAY -- deferred, deferred_sequential_split
# =========================================================================

con = xo.connect()
data = con.register(_load_data(), "sparse_signal")

# Sequential split (first ~67% train, last ~33% test)
train_data, test_data = deferred_sequential_split(
    data,
    features=FEATURE_COLS,
    target=TARGET_COL,
    order_by=ROW_IDX,
)
name_to_xorq_exprs = {
    name: _deferred_fit_xorq_pipeline(sklearn_pipeline, train_data, test_data)
    for name, sklearn_pipeline in name_to_pipeline.items()
}
# expose the exprs in the script to invoke `xorq build plot_lasso_and_elasticnet.py --expr $expr_name`
(xorq_lasso_preds, xorq_ard_preds, xorq_elastic_net_preds) = (
    name_to_xorq_exprs[name]["preds"]
    for name in methods
)


# =========================================================================
# Run and plot side by side
# =========================================================================


def compare_results(sk_coefs, sk_r2_scores, xo_coefs, xo_r2_scores):
    # ---- Compare results ----
    # Note: Results may differ slightly due to subtle differences in how
    # TimeSeriesSplit and train_test_split handle the splits. Both methods
    # preserve temporal order and use roughly the same train/test proportions,
    # but TimeSeriesSplit uses fold-based splitting while train_test_split
    # uses a simple cutoff.
    print("\n=== Comparing Results ===")
    for name in methods:
        sk_r2, xo_r2 = (dct[name] for dct in (sk_r2_scores, xo_r2_scores))
        r2_diff = abs(sk_r2 - xo_r2)
        print(
            f"{name} R^2 - sklearn: {sk_r2:.3f}, xorq: {xo_r2:.3f}, diff: {r2_diff:.4f}"
        )

        sk_coef, xo_coef = (dct[name] for dct in (sk_coefs, xo_coefs))
        coef_diff = np.max(np.abs(sk_coef - xo_coef))
        print(f"{name} max coef difference: {coef_diff:.6f}")


def save_comparison_plot(true_coef, sk_coefs, sk_r2_scores, xo_coefs, xo_r2_scores):
    # Build coefficient matrices
    row_labels = ["True coefficients", *methods]
    sk_coef_matrix, xo_coef_matrix = (
        np.vstack(
            [
                true_coef,
                *(
                    coefs[method] for method in methods
                ),
            ]
        )
        for coefs in (sk_coefs, xo_coefs)
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
    out = "imgs/lasso_and_elasticnet.png"
    save_fig(out, fig, bbox_inches=None)


def main():
    df = _load_data()
    true_coef = df.attrs["true_coef"]

    print("=== SKLEARN WAY ===")
    sk_r2_scores, sk_coefs, _ = compute_with_sklearn(name_to_pipeline, df)

    # Execute deferred expressions
    print("\n=== XORQ WAY ===")
    xo_r2_scores, xo_coefs = compute_with_xorq(name_to_xorq_exprs)

    compare_results(sk_coefs, sk_r2_scores, xo_coefs, xo_r2_scores)

    print(
        "\nBoth approaches produce L1-regularized sparse models with similar sparsity patterns."
    )
    save_comparison_plot(true_coef, sk_coefs, sk_r2_scores, xo_coefs, xo_r2_scores)


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
