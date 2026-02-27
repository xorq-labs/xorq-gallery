"""L1-based models for Sparse Signals
====================================

sklearn: Generate synthetic sparse signals with correlated features, fit Lasso,
ARDRegression, and ElasticNet models eagerly on numpy arrays, evaluate R^2 score
and compare estimated coefficients with ground-truth. Split with a row-index
cutoff to preserve temporal order.

xorq: Same models wrapped in Pipeline.from_instance. Data is an ibis expression,
split via row-index filter on the expression graph, fit/predict deferred, metrics
via deferred_sklearn_metric, coefficients extracted from the fitted pipeline.

Both produce identical R^2 scores and coefficient patterns.

Dataset: Synthetic sparse sinusoidal signals with Gaussian noise
"""

from __future__ import annotations

import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm
from sklearn.base import clone
from sklearn.linear_model import ARDRegression, ElasticNet, Lasso
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.sklearn.sklearn_lib import SklearnXorqComparator, assert_results
from xorq_gallery.utils import fig_to_image


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

# TODO(xorq): train_test_splits needs shuffle=False mode for full match;
# until then we use a manual row_idx cutoff to guarantee identical splits.
TEST_SIZE = 0.25
CUTOFF = int(N_SAMPLES * (1 - TEST_SIZE))  # row_idx < CUTOFF -> train


# ---------------------------------------------------------------------------
# Data generation
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


def _load_data():
    """Load data as pandas DataFrame with row_idx for temporal ordering."""
    true_coef, X, y, time_step = _generate_sparse_signal()

    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df[TARGET_COL] = y
    df["time_step"] = time_step
    df[ROW_IDX] = range(len(df))

    # Store true_coef separately for comparison
    df.attrs["true_coef"] = true_coef

    return df


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------


def _build_coefficient_heatmap(coef_matrix, row_labels, r2_scores, title_prefix):
    """Build coefficient comparison heatmap using matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        coef_matrix,
        aspect="auto",
        cmap="seismic_r",
        norm=SymLogNorm(linthresh=10e-4, vmin=-1, vmax=1),
    )

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_ylabel("linear model")
    ax.set_xlabel("coefficients")

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


# ---------------------------------------------------------------------------
# build_exprs: the full xorq expression chain including split
# ---------------------------------------------------------------------------


def build_exprs(sklearn_pipeline, input_expr, features, target, pred_name):
    """Split + Pipeline.from_instance + fit + predict + r2 metric.

    The split is part of the expression graph: filter on row_idx cutoff.
    """
    train_expr = input_expr.filter(input_expr[ROW_IDX] < CUTOFF)
    test_expr = input_expr.filter(input_expr[ROW_IDX] >= CUTOFF)

    pipeline = Pipeline.from_instance(sklearn_pipeline)
    fitted = pipeline.fit(train_expr, features=features, target=target)
    preds = fitted.predict(test_expr, name=pred_name)
    return {
        "fitted_pipeline": fitted,
        "preds": preds,
        "metrics": preds.agg(
            r2=deferred_sklearn_metric(
                target=target,
                pred=pred_name,
                metric=r2_score,
            ),
        ),
    }


# ---------------------------------------------------------------------------
# compute_sklearn: pure sklearn, no xorq imports
# ---------------------------------------------------------------------------


def compute_sklearn(df, pipelines):
    """Eager sklearn: time-ordered split, fit three L1-based models, score.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset with row_idx column.
    pipelines : tuple[tuple[str, SklearnPipeline], ...]
        Named sklearn pipelines to fit.
    """
    train_df = df[df[ROW_IDX] < CUTOFF]
    test_df = df[df[ROW_IDX] >= CUTOFF]
    X_train = train_df[list(FEATURE_COLS)].values
    X_test = test_df[list(FEATURE_COLS)].values
    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    results = {}
    for name, pipe_template in pipelines:
        t0 = time()
        pipe = clone(pipe_template)
        fitted = pipe.fit(X_train, y_train)
        fit_time = time() - t0
        preds = fitted.predict(X_test)
        r2 = r2_score(y_test, preds)
        coef = fitted[-1].coef_
        print(f"  sklearn {name}: R^2 = {r2:.3f}, fit time = {fit_time:.3f}s")
        results[name] = {"r2": r2, "coef": coef, "fit_time": fit_time}
    return results


# ---------------------------------------------------------------------------
# compute_xorq: execute deferred expressions
# ---------------------------------------------------------------------------


def compute_xorq(deferred_exprs):
    """Execute deferred predictions, metrics, and coef extraction.

    Parameters
    ----------
    deferred_exprs : dict[str, dict[str, Expr]]
        ``{pipeline_name: {"preds": expr, "metrics": expr, ...}, ...}``
    """
    results = {}
    for name, e in deferred_exprs.items():
        r2 = e["metrics"].execute()["r2"].iloc[0]
        coef = e["fitted_pipeline"].fitted_steps[-1].model.coef_
        print(f"  xorq   {name}: R^2 = {r2:.3f}")
        results[name] = {"r2": r2, "coef": coef}
    return results


# ---------------------------------------------------------------------------
# build_assertions: what gets compared
# ---------------------------------------------------------------------------


PIPELINE_NAMES = ("Lasso", "ARD", "ElasticNet")


def build_assertions(sk, xo):
    """Build assertion pairs for R^2 scores and coefficients."""
    sk_r2_df = pd.DataFrame({n: [sk[n]["r2"]] for n in PIPELINE_NAMES}, index=["r2"])
    xo_r2_df = pd.DataFrame({n: [xo[n]["r2"]] for n in PIPELINE_NAMES}, index=["r2"])
    sk_coef_df = pd.DataFrame({n: sk[n]["coef"] for n in PIPELINE_NAMES})
    xo_coef_df = pd.DataFrame({n: xo[n]["coef"] for n in PIPELINE_NAMES})
    return [
        ("R^2 scores", sk_r2_df, xo_r2_df),
        ("Coefficients", sk_coef_df, xo_coef_df),
    ]


# ---------------------------------------------------------------------------
# plot: composite figure
# ---------------------------------------------------------------------------


def plot(sk, xo, df):
    """Build coefficient heatmaps and composite plot."""
    true_coef = df.attrs["true_coef"]
    row_labels = ["True coefficients", *PIPELINE_NAMES]

    sk_fig = _build_coefficient_heatmap(
        np.vstack([true_coef, *(sk[n]["coef"] for n in PIPELINE_NAMES)]),
        row_labels,
        {n: sk[n]["r2"] for n in PIPELINE_NAMES},
        "sklearn",
    )
    xo_fig = _build_coefficient_heatmap(
        np.vstack([true_coef, *(xo[n]["coef"] for n in PIPELINE_NAMES)]),
        row_labels,
        {n: xo[n]["r2"] for n in PIPELINE_NAMES},
        "xorq",
    )

    # Composite: sklearn (left) | xorq (right)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].axis("off")

    fig.suptitle("L1-based models for Sparse Signals: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    os.makedirs("imgs", exist_ok=True)
    out = "imgs/lasso_and_elasticnet.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {out}")


# ---------------------------------------------------------------------------
# Shared sklearn pipelines
# ---------------------------------------------------------------------------

SHARED_SKLEARN_PIPELINES = (
    ("Lasso", SklearnPipeline([("lasso", Lasso(alpha=LASSO_ALPHA))])),
    ("ARD", SklearnPipeline([("ard", ARDRegression())])),
    (
        "ElasticNet",
        SklearnPipeline(
            [
                (
                    "elasticnet",
                    ElasticNet(
                        alpha=ELASTICNET_ALPHA,
                        l1_ratio=ELASTICNET_L1_RATIO,
                    ),
                )
            ]
        ),
    ),
)


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------

comparator = SklearnXorqComparator(
    name="lasso_and_elasticnet",
    named_pipelines=SHARED_SKLEARN_PIPELINES,
    df=_load_data(),
    features=FEATURE_COLS,
    build_exprs_fn=build_exprs,
    target=TARGET_COL,
    pred_col=PRED_COL,
    metrics=(("r2", r2_score),),
)

# -- Module-level exprs (for xorq build --expr) ----------------------------
xorq_exprs = comparator.deferred_exprs

xorq_lasso_fitted_pipeline = xorq_exprs["Lasso"]["fitted_pipeline"]
xorq_lasso_preds = xorq_exprs["Lasso"]["preds"]
xorq_lasso_metrics = xorq_exprs["Lasso"]["metrics"]
xorq_ard_fitted_pipeline = xorq_exprs["ARD"]["fitted_pipeline"]
xorq_ard_preds = xorq_exprs["ARD"]["preds"]
xorq_ard_metrics = xorq_exprs["ARD"]["metrics"]
xorq_elasticnet_fitted_pipeline = xorq_exprs["ElasticNet"]["fitted_pipeline"]
xorq_elasticnet_preds = xorq_exprs["ElasticNet"]["preds"]
xorq_elasticnet_metrics = xorq_exprs["ElasticNet"]["metrics"]


# =========================================================================
# Run
# =========================================================================


def main():
    print("=== SKLEARN WAY ===")
    sk = compute_sklearn(comparator.df, SHARED_SKLEARN_PIPELINES)

    print("\n=== XORQ WAY ===")
    xo = compute_xorq(comparator.deferred_exprs)

    assert_results(build_assertions(sk, xo))
    plot(sk, xo, comparator.df)


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
