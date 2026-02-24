"""Categorical Feature Support in Gradient Boosting
================================================

sklearn: Compare HistGradientBoostingRegressor training times and prediction
performances with different categorical encoding strategies (dropped, one-hot,
ordinal, target, native). Evaluate using cross-validation with 5 folds,
measuring mean absolute percentage error and fit times. Test both full-depth
and limited-depth (underfitting) models.

xorq: Same pipelines wrapped in Pipeline.from_instance, deferred cross-validation
via deferred_cross_val_score with KFold(n_splits=5). Per-fold MAPE scores match
sklearn exactly.

Both produce identical cross-validation scores.

Dataset: Ames Housing (OpenML 42165)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xorq.api as xo
from sklearn.compose import make_column_transformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder
from xorq.expr.ml.cross_validation import (
    apply_deterministic_sort,
    deferred_cross_val_score,
)
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    fig_to_image,
)


# ---------------------------------------------------------------------------
# Feature groups and constants
# ---------------------------------------------------------------------------

CATEGORICAL_COLUMNS = (
    "BldgType",
    "GarageFinish",
    "LotConfig",
    "Functional",
    "MasVnrType",
    "HouseStyle",
    "FireplaceQu",
    "ExterCond",
    "ExterQual",
    "PoolQC",
)

NUMERICAL_COLUMNS = (
    "ThreeSsnPorch",
    "Fireplaces",
    "BsmtHalfBath",
    "HalfBath",
    "GarageCars",
    "TotRmsAbvGrd",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "GrLivArea",
    "ScreenPorch",
)

ALL_FEATURE_COLS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
TARGET_COL = "SalePrice"
ROW_IDX = "row_idx"
RANDOM_STATE = 42
N_SPLITS = 5

CV_SPLITTER = KFold(n_splits=N_SPLITS, shuffle=False)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data():
    """Load Ames Housing dataset from OpenML."""
    X, y = fetch_openml(data_id=42165, as_frame=True, return_X_y=True)

    # Rename "3SsnPorch" for valid Python identifier
    X = X.rename(columns={"3SsnPorch": "ThreeSsnPorch"})

    X = X[list(ALL_FEATURE_COLS)]
    X[list(CATEGORICAL_COLUMNS)] = X[list(CATEGORICAL_COLUMNS)].astype("category")

    df = X.copy()
    df[TARGET_COL] = y
    df[ROW_IDX] = range(len(df))

    n_cat = X.select_dtypes(include="category").shape[1]
    n_num = X.select_dtypes(include="number").shape[1]
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of categorical features: {n_cat}")
    print(f"Number of numerical features: {n_num}")

    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_performance_tradeoff(results, title):
    """Scatter plot comparing fit time vs MAPE across encoding schemes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["s", "o", "^", "x", "D"]

    for idx, (name, result) in enumerate(results):
        test_error = -result["test_score"]
        mean_fit_time = np.mean(result["fit_time"])
        mean_score = np.mean(test_error)
        std_fit_time = np.std(result["fit_time"])
        std_score = np.std(test_error)

        ax.scatter(result["fit_time"], test_error, label=name, marker=markers[idx])
        ax.scatter(mean_fit_time, mean_score, color="k", marker=markers[idx])
        ax.errorbar(x=mean_fit_time, y=mean_score, yerr=std_score, c="k", capsize=2)
        ax.errorbar(x=mean_fit_time, y=mean_score, xerr=std_fit_time, c="k", capsize=2)

    ax.set_xscale("log")
    nticks = 7
    x0, x1 = np.log10(ax.get_xlim())
    ticks = np.logspace(x0, x1, nticks)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%1.1e"))
    ax.minorticks_off()

    ax.annotate(
        "  best\nmodels",
        xy=(0.04, 0.04),
        xycoords="axes fraction",
        xytext=(0.09, 0.14),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.5},
    )
    ax.set_xlabel("Time to fit (seconds)")
    ax.set_ylabel("Mean Absolute Percentage Error")
    ax.set_title(title)
    ax.legend()

    return fig


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def _build_pipelines(max_depth=None, max_iter=100):
    """Build preprocessing pipelines for different encoding strategies."""
    pipelines = {}

    # Dropped
    dropper = make_column_transformer(
        ("drop", CATEGORICAL_COLUMNS), remainder="passthrough"
    )
    pipelines["Dropped"] = SklearnPipeline(
        [
            ("columntransformer", dropper),
            (
                "histgradientboostingregressor",
                HistGradientBoostingRegressor(
                    random_state=RANDOM_STATE, max_depth=max_depth, max_iter=max_iter
                ),
            ),
        ]
    )

    # One-hot
    one_hot = make_column_transformer(
        (
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            CATEGORICAL_COLUMNS,
        ),
        remainder="passthrough",
    )
    pipelines["One Hot"] = SklearnPipeline(
        [
            ("columntransformer", one_hot),
            (
                "histgradientboostingregressor",
                HistGradientBoostingRegressor(
                    random_state=RANDOM_STATE, max_depth=max_depth, max_iter=max_iter
                ),
            ),
        ]
    )

    # Ordinal
    ordinal = make_column_transformer(
        (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
            CATEGORICAL_COLUMNS,
        ),
        remainder="passthrough",
    )
    pipelines["Ordinal"] = SklearnPipeline(
        [
            ("columntransformer", ordinal),
            (
                "histgradientboostingregressor",
                HistGradientBoostingRegressor(
                    random_state=RANDOM_STATE, max_depth=max_depth, max_iter=max_iter
                ),
            ),
        ]
    )

    # Target encoding
    target_enc = make_column_transformer(
        (
            TargetEncoder(target_type="continuous", random_state=RANDOM_STATE),
            CATEGORICAL_COLUMNS,
        ),
        remainder="passthrough",
    )
    pipelines["Target"] = SklearnPipeline(
        [
            ("columntransformer", target_enc),
            (
                "histgradientboostingregressor",
                HistGradientBoostingRegressor(
                    random_state=RANDOM_STATE, max_depth=max_depth, max_iter=max_iter
                ),
            ),
        ]
    )

    # Native categorical support (sklearn only)
    pipelines["Native"] = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        categorical_features="from_dtype",
        max_depth=max_depth,
        max_iter=max_iter,
    )

    return pipelines


# Encoders that work with xorq (Native requires sklearn-specific handling)
XORQ_ENCODER_NAMES = ("Dropped", "One Hot", "Ordinal", "Target")


# =========================================================================
# SKLEARN WAY -- eager cross_validate
# =========================================================================


def sklearn_way(df_sorted, pipelines):
    """Eager sklearn: cross-validate each pipeline with KFold.

    df_sorted must be sorted by apply_deterministic_sort so fold assignments
    match the xorq side.
    """
    X = df_sorted[list(ALL_FEATURE_COLS)]
    y = df_sorted[TARGET_COL]

    results = [
        (
            name,
            cross_validate(
                pipelines[name],
                X,
                y,
                cv=CV_SPLITTER,
                scoring="neg_mean_absolute_percentage_error",
                n_jobs=-1,
            ),
        )
        for name in list(pipelines.keys())
    ]

    for name, result in results:
        mean_mape = -np.mean(result["test_score"])
        std_mape = np.std(-result["test_score"])
        print(f"  sklearn {name:12s}: MAPE={mean_mape:.4f} (+/-{std_mape:.4f})")

    return results


# =========================================================================
# XORQ WAY -- deferred cross-validation
# =========================================================================


def xorq_way(data, pipelines):
    """Deferred xorq: deferred_cross_val_score per encoding pipeline.

    Returns dict of CrossValScore objects keyed by encoder name.
    Nothing is executed until .execute().
    """
    results = {}
    for name in XORQ_ENCODER_NAMES:
        xorq_pipe = Pipeline.from_instance(pipelines[name])
        cv_result = deferred_cross_val_score(
            xorq_pipe,
            data,
            ALL_FEATURE_COLS,
            TARGET_COL,
            cv=CV_SPLITTER,
            random_seed=RANDOM_STATE,
        )
        results[name] = cv_result

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    print("\n" + "=" * 70)
    print("Full-depth models (default)")
    print("=" * 70)

    df = _load_data()

    con = xo.connect()
    table = con.register(df, "ames_housing")

    # Sort by deterministic hash so sklearn KFold sees same row order as xorq
    df_sorted = apply_deterministic_sort(table, random_seed=RANDOM_STATE).execute()

    print("\n=== SKLEARN WAY (full-depth) ===")
    pipelines_full = _build_pipelines()
    sk_results_full = sklearn_way(df_sorted, pipelines_full)

    print("\n=== XORQ WAY (full-depth) ===")
    xo_pipelines_full = _build_pipelines()
    xo_deferred_full = xorq_way(table, xo_pipelines_full)

    # Execute deferred CV scores
    xo_scores_full = {}
    for name in XORQ_ENCODER_NAMES:
        scores = xo_deferred_full[name].execute()
        xo_scores_full[name] = scores
        print(f"  xorq   {name:12s}: MAPE={scores.mean():.4f} (+/-{scores.std():.4f})")

    # Assert: per-fold scores match
    print("\n=== ASSERTIONS (full-depth) ===")
    for name, sk_result in sk_results_full:
        if name == "Native":
            print(f"  {name:12s}: skipped (not supported in xorq)")
            continue
        sk_scores = -sk_result["test_score"]
        xo_scores = xo_scores_full[name]
        np.testing.assert_allclose(sk_scores, xo_scores, rtol=1e-6)
        print(f"  {name:12s}: per-fold scores match")
    print("Assertions passed.")

    sk_fig_full = _plot_performance_tradeoff(
        sk_results_full, "sklearn - Gradient Boosting on Ames Housing"
    )

    print("\n" + "=" * 70)
    print("Limited-depth models (max_depth=3, max_iter=15)")
    print("=" * 70)

    print("\n=== SKLEARN WAY (underfit) ===")
    pipelines_underfit = _build_pipelines(max_depth=3, max_iter=15)
    sk_results_underfit = sklearn_way(df_sorted, pipelines_underfit)

    print("\n=== XORQ WAY (underfit) ===")
    xo_pipelines_underfit = _build_pipelines(max_depth=3, max_iter=15)
    xo_deferred_underfit = xorq_way(table, xo_pipelines_underfit)

    xo_scores_underfit = {}
    for name in XORQ_ENCODER_NAMES:
        scores = xo_deferred_underfit[name].execute()
        xo_scores_underfit[name] = scores
        print(f"  xorq   {name:12s}: MAPE={scores.mean():.4f} (+/-{scores.std():.4f})")

    print("\n=== ASSERTIONS (underfit) ===")
    for name, sk_result in sk_results_underfit:
        if name == "Native":
            print(f"  {name:12s}: skipped (not supported in xorq)")
            continue
        sk_scores = -sk_result["test_score"]
        xo_scores = xo_scores_underfit[name]
        np.testing.assert_allclose(sk_scores, xo_scores, rtol=1e-6)
        print(f"  {name:12s}: per-fold scores match")
    print("Assertions passed.")

    sk_fig_underfit = _plot_performance_tradeoff(
        sk_results_underfit,
        "sklearn - Gradient Boosting on Ames Housing (few and shallow trees)",
    )

    # Composite plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    axes[0].imshow(fig_to_image(sk_fig_full))
    axes[0].axis("off")
    axes[0].set_title("Full-depth models", fontsize=12, pad=10)

    axes[1].imshow(fig_to_image(sk_fig_underfit))
    axes[1].axis("off")
    axes[1].set_title(
        "Limited-depth models (max_depth=3, max_iter=15)", fontsize=12, pad=10
    )

    fig.suptitle(
        "Categorical Feature Support in Gradient Boosting: sklearn",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout()
    out = "imgs/plot_gradient_boosting_categorical.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
