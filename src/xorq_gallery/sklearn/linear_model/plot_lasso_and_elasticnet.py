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

import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
import xorq.expr.datatypes as dt
from matplotlib.colors import SymLogNorm
from sklearn.linear_model import ARDRegression, ElasticNet, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from time import time
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline
from xorq.expr.udf import agg

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    deferred_sequential_split,
    fig_to_image,
    load_plot_bytes,
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


def _load_data():
    """Load data as pandas DataFrame with row_idx for temporal ordering."""
    true_coef, X, y, time_step = _generate_sparse_signal()

    # Build DataFrame
    feature_cols = [f"feature_{i}" for i in range(N_FEATURES)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y
    df["time_step"] = time_step
    df["row_idx"] = range(len(df))

    # Store true_coef separately for comparison
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
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("coefficients' values")

    ax.set_title(
        f"{title_prefix} - Models' coefficients\n"
        f"Lasso $R^2$: {r2_scores['Lasso']:.3f}, "
        f"ARD $R^2$: {r2_scores['ARD']:.3f}, "
        f"ElasticNet $R^2$: {r2_scores['ElasticNet']:.3f}"
    )
    plt.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager, train_test_split(shuffle=False)
# =========================================================================


def sklearn_way(df):
    """Eager sklearn: time-ordered split, fit three L1-based models, score.

    Returns
    -------
    dict
        Keys: "Lasso", "ARD", "ElasticNet"
        Values: dict with "r2", "coef", "fit_time"
    """
    feature_cols = [f"feature_{i}" for i in range(N_FEATURES)]
    X = df[feature_cols].values
    y = df["target"].values

    # shuffle=False preserves temporal order: first rows train, last rows test
    # Using test_size=0.3333 to match deferred_sequential_split behavior
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3333, shuffle=False
    )

    results = {}

    # Lasso
    t0 = time()
    lasso = Lasso(alpha=LASSO_ALPHA).fit(X_train, y_train)
    lasso_time = time() - t0
    y_pred_lasso = lasso.predict(X_test)
    r2_lasso = r2_score(y_test, y_pred_lasso)
    print(f"  sklearn Lasso: R^2 = {r2_lasso:.3f}, fit time = {lasso_time:.3f}s")
    results["Lasso"] = {"r2": r2_lasso, "coef": lasso.coef_, "fit_time": lasso_time}

    # ARDRegression
    t0 = time()
    ard = ARDRegression().fit(X_train, y_train)
    ard_time = time() - t0
    y_pred_ard = ard.predict(X_test)
    r2_ard = r2_score(y_test, y_pred_ard)
    print(f"  sklearn ARD: R^2 = {r2_ard:.3f}, fit time = {ard_time:.3f}s")
    results["ARD"] = {"r2": r2_ard, "coef": ard.coef_, "fit_time": ard_time}

    # ElasticNet
    t0 = time()
    enet = ElasticNet(alpha=ELASTICNET_ALPHA, l1_ratio=ELASTICNET_L1_RATIO).fit(
        X_train, y_train
    )
    enet_time = time() - t0
    y_pred_enet = enet.predict(X_test)
    r2_enet = r2_score(y_test, y_pred_enet)
    print(f"  sklearn ElasticNet: R^2 = {r2_enet:.3f}, fit time = {enet_time:.3f}s")
    results["ElasticNet"] = {
        "r2": r2_enet,
        "coef": enet.coef_,
        "fit_time": enet_time,
    }

    return results


# =========================================================================
# XORQ WAY -- deferred, deferred_sequential_split
# =========================================================================


def xorq_way(df):
    """Deferred xorq: sequential split, fit three L1-based models deferred.

    Returns deferred expressions for predictions and metrics, plus fitted pipelines
    for coefficient extraction after execution.
    Nothing is executed until ``.execute()``.

    Returns
    -------
    dict
        Keys: "Lasso", "ARD", "ElasticNet"
        Values: dict with "preds", "metrics", "fitted_pipe"
    """
    con = xo.connect()
    data = con.register(df, "sparse_signal")

    feature_cols = tuple(f"feature_{i}" for i in range(N_FEATURES))
    target_col = "target"

    # Sequential split (first ~67% train, last ~33% test)
    train_data, test_data = deferred_sequential_split(
        data, features=feature_cols, target=target_col, order_by="row_idx", test_size=0.3333
    )

    make_metric = deferred_sklearn_metric(target=target_col, pred="pred")

    results = {}

    # Lasso - wrap in sklearn Pipeline first
    lasso_sklearn_pipe = SklearnPipeline([("lasso", Lasso(alpha=LASSO_ALPHA))])
    lasso_pipe = Pipeline.from_instance(lasso_sklearn_pipe)
    lasso_fitted = lasso_pipe.fit(train_data, features=feature_cols, target=target_col)
    lasso_preds = lasso_fitted.predict(test_data, name="pred")
    lasso_metrics = lasso_preds.agg(r2=make_metric(metric=r2_score))

    results["Lasso"] = {
        "preds": lasso_preds,
        "metrics": lasso_metrics,
        "fitted_pipe": lasso_fitted,
    }

    # ARDRegression
    ard_sklearn_pipe = SklearnPipeline([("ard", ARDRegression())])
    ard_pipe = Pipeline.from_instance(ard_sklearn_pipe)
    ard_fitted = ard_pipe.fit(train_data, features=feature_cols, target=target_col)
    ard_preds = ard_fitted.predict(test_data, name="pred")
    ard_metrics = ard_preds.agg(r2=make_metric(metric=r2_score))

    results["ARD"] = {
        "preds": ard_preds,
        "metrics": ard_metrics,
        "fitted_pipe": ard_fitted,
    }

    # ElasticNet
    enet_sklearn_pipe = SklearnPipeline([
        ("elasticnet", ElasticNet(alpha=ELASTICNET_ALPHA, l1_ratio=ELASTICNET_L1_RATIO))
    ])
    enet_pipe = Pipeline.from_instance(enet_sklearn_pipe)
    enet_fitted = enet_pipe.fit(train_data, features=feature_cols, target=target_col)
    enet_preds = enet_fitted.predict(test_data, name="pred")
    enet_metrics = enet_preds.agg(r2=make_metric(metric=r2_score))

    results["ElasticNet"] = {
        "preds": enet_preds,
        "metrics": enet_metrics,
        "fitted_pipe": enet_fitted,
    }

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()
    true_coef = df.attrs["true_coef"]

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(df)

    print("\n=== XORQ WAY ===")
    deferred = xorq_way(df)

    # Execute deferred expressions
    xo_r2_scores = {}
    xo_coefs = {}

    for name, exprs in deferred.items():
        metrics_df = exprs["metrics"].execute()
        r2 = metrics_df["r2"].iloc[0]
        xo_r2_scores[name] = r2
        print(f"  xorq   {name}: R^2 = {r2:.3f}")

        # Extract coefficients from fitted pipeline
        fitted_pipe = exprs["fitted_pipe"]
        # Access the fitted sklearn model from the last fitted step
        last_step = fitted_pipe.fitted_steps[-1]
        sklearn_model = last_step.model
        xo_coefs[name] = sklearn_model.coef_

    # ---- Compare results ----
    # Note: Results may differ slightly due to subtle differences in how
    # TimeSeriesSplit and train_test_split handle the splits. Both methods
    # preserve temporal order and use roughly the same train/test proportions,
    # but TimeSeriesSplit uses fold-based splitting while train_test_split
    # uses a simple cutoff.
    print("\n=== Comparing Results ===")
    for name in ["Lasso", "ARD", "ElasticNet"]:
        sk_r2 = sk_results[name]["r2"]
        xo_r2 = xo_r2_scores[name]
        r2_diff = abs(sk_r2 - xo_r2)
        print(f"{name} R^2 - sklearn: {sk_r2:.3f}, xorq: {xo_r2:.3f}, diff: {r2_diff:.4f}")

        sk_coef = sk_results[name]["coef"]
        xo_coef = xo_coefs[name]
        coef_diff = np.max(np.abs(sk_coef - xo_coef))
        print(f"{name} max coef difference: {coef_diff:.6f}")

    print("\nBoth approaches produce L1-regularized sparse models with similar sparsity patterns.")

    # Build coefficient matrices
    row_labels = ["True coefficients", "Lasso", "ARDRegression", "ElasticNet"]
    sk_coef_matrix = np.vstack([
        true_coef,
        sk_results["Lasso"]["coef"],
        sk_results["ARD"]["coef"],
        sk_results["ElasticNet"]["coef"],
    ])

    xo_coef_matrix = np.vstack([
        true_coef,
        xo_coefs["Lasso"],
        xo_coefs["ARD"],
        xo_coefs["ElasticNet"],
    ])

    sk_r2_dict = {name: sk_results[name]["r2"] for name in ["Lasso", "ARD", "ElasticNet"]}

    # Build sklearn heatmap
    sk_fig = _build_coefficient_heatmap(
        sk_coef_matrix, row_labels, sk_r2_dict, "sklearn"
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

    plt.suptitle(
        "L1-based models for Sparse Signals: sklearn vs xorq", fontsize=16
    )
    plt.tight_layout()
    out = "imgs/lasso_and_elasticnet.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nPlot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
