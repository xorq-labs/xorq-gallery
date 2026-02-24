"""Comparing Target Encoder with Other Encoders
============================================

sklearn: Build pipelines with HistGradientBoostingRegressor using different
categorical encoding strategies (drop, ordinal, one-hot, target, mixed). Evaluate
using cross_validate with 3-fold CV. Compare RMSE across encoding schemes.

xorq: Same pipelines wrapped in Pipeline.from_instance, deferred cross-validation
via deferred_cross_val_score with KFold(n_splits=3). Per-fold test scores match
sklearn exactly.

Both produce identical cross-validation scores.

Dataset: Wine Reviews (OpenML 42074)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.compose import ColumnTransformer
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

NUMERICAL_FEATURES = ("price",)
CATEGORICAL_FEATURES = (
    "country",
    "province",
    "region_1",
    "region_2",
    "variety",
    "winery",
)
ALL_FEATURE_COLS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "points"
ROW_IDX = "row_idx"
RANDOM_STATE = 0
N_CV_FOLDS = 3

CV_SPLITTER = KFold(n_splits=N_CV_FOLDS, shuffle=False)

# Encoders that work with xorq (mixed_target requires sklearn-specific handling)
XORQ_ENCODER_NAMES = ("drop", "ordinal", "one_hot", "target")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data():
    """Load Wine Reviews dataset from OpenML."""
    wine_reviews = fetch_openml(data_id=42074, as_frame=True, parser="pandas")
    df = wine_reviews.frame
    df[ROW_IDX] = range(len(df))
    return df


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def _build_pipelines():
    """Build preprocessing pipelines for different encoding strategies."""
    max_iter = 20

    categorical_preprocessors = (
        ("drop", "drop"),
        (
            "ordinal",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
        (
            "one_hot",
            OneHotEncoder(
                handle_unknown="ignore", max_categories=20, sparse_output=False
            ),
        ),
        ("target", TargetEncoder(target_type="continuous")),
    )

    pipelines = {
        name: SklearnPipeline(
            [
                (
                    "columntransformer",
                    ColumnTransformer(
                        [
                            ("numerical", "passthrough", list(NUMERICAL_FEATURES)),
                            (
                                "categorical",
                                categorical_preprocessor,
                                list(CATEGORICAL_FEATURES),
                            ),
                        ]
                    ),
                ),
                (
                    "histgradientboostingregressor",
                    HistGradientBoostingRegressor(
                        random_state=RANDOM_STATE, max_iter=max_iter
                    ),
                ),
            ]
        )
        for name, categorical_preprocessor in categorical_preprocessors
    }

    # Mixed encoding: high cardinality -> target, low cardinality -> ordinal
    df_sample = _load_data()
    n_unique = df_sample[CATEGORICAL_FEATURES].nunique()
    high_card = n_unique[n_unique > 255].index.tolist()
    low_card = n_unique[n_unique <= 255].index.tolist()

    mixed_preprocessor = ColumnTransformer(
        [
            ("numerical", "passthrough", list(NUMERICAL_FEATURES)),
            ("high_cardinality", TargetEncoder(target_type="continuous"), high_card),
            (
                "low_cardinality",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                low_card,
            ),
        ],
        verbose_feature_names_out=False,
    )
    mixed_preprocessor.set_output(transform="pandas")
    pipelines["mixed_target"] = SklearnPipeline(
        [
            ("columntransformer", mixed_preprocessor),
            (
                "histgradientboostingregressor",
                HistGradientBoostingRegressor(
                    random_state=RANDOM_STATE,
                    max_iter=max_iter,
                    categorical_features=low_card,
                ),
            ),
        ]
    )

    return pipelines


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _build_results_plot(results_df, title):
    """Bar plot comparing RMSE across encoding schemes."""
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        sharey=True,
        constrained_layout=True,
    )

    xticks = range(len(results_df))
    name_to_color = dict(zip(results_df.index, ["C0", "C1", "C2", "C3", "C4"]))

    for subset, ax in zip(["test", "train"], [ax1, ax2]):
        mean_col = f"rmse_{subset}_mean"
        std_col = f"rmse_{subset}_std"
        data = results_df[[mean_col, std_col]].sort_values(mean_col)
        ax.bar(
            x=xticks,
            height=data[mean_col],
            yerr=data[std_col],
            width=0.9,
            color=[name_to_color.get(name, "C0") for name in data.index],
        )
        ax.set(
            title=f"RMSE ({subset.title()})",
            xlabel="Encoding Scheme",
            xticks=xticks,
            xticklabels=data.index,
        )

    fig.suptitle(title, fontsize=14)
    return fig


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

    results = []
    for name, pipe in pipelines.items():
        result = cross_validate(
            pipe,
            X,
            y,
            scoring="neg_root_mean_squared_error",
            cv=CV_SPLITTER,
            return_train_score=True,
        )
        rmse_test = -result["test_score"]
        rmse_train = -result["train_score"]
        results.append(
            {
                "preprocessor": name,
                "rmse_test_mean": rmse_test.mean(),
                "rmse_test_std": rmse_test.std(),
                "rmse_train_mean": rmse_train.mean(),
                "rmse_train_std": rmse_train.std(),
                "test_scores": rmse_test,
            }
        )
        print(
            f"  sklearn {name:15s}: RMSE test={rmse_test.mean():.4f}, train={rmse_train.mean():.4f}"
        )

    results_df = (
        pd.DataFrame(results).set_index("preprocessor").sort_values("rmse_test_mean")
    )
    return results_df


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
            scoring="neg_root_mean_squared_error",
            random_seed=RANDOM_STATE,
        )
        results[name] = cv_result

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()

    con = xo.connect()
    table = con.register(df, "wine_reviews")

    # Sort by deterministic hash so sklearn KFold sees same row order as xorq
    df_sorted = apply_deterministic_sort(table, random_seed=RANDOM_STATE).execute()

    print("=== SKLEARN WAY ===")
    pipelines = _build_pipelines()
    sk_results = sklearn_way(df_sorted, pipelines)

    print("\n=== XORQ WAY ===")
    xo_pipelines = _build_pipelines()
    xo_deferred = xorq_way(table, xo_pipelines)

    # Execute deferred CV scores
    xo_scores = {}
    for name in XORQ_ENCODER_NAMES:
        scores = xo_deferred[name].execute()
        xo_scores[name] = scores
        # deferred_cross_val_score returns raw scorer output (negative RMSE)
        rmse = -scores
        print(f"  xorq   {name:15s}: RMSE test={rmse.mean():.4f} (+/-{rmse.std():.4f})")

    # Assert: per-fold test scores match
    print("\n=== ASSERTIONS ===")
    for name in XORQ_ENCODER_NAMES:
        sk_test_scores = sk_results.loc[name, "test_scores"]
        xo_test_scores = -xo_scores[name]  # negate to get positive RMSE
        np.testing.assert_allclose(sk_test_scores, xo_test_scores, rtol=1e-6)
        print(f"  {name:15s}: per-fold test scores match")
    print("Assertions passed.")

    # Build plots
    # xorq results as DataFrame (test only, no train scores from deferred CV)
    xo_results = []
    for name in XORQ_ENCODER_NAMES:
        rmse = -xo_scores[name]
        xo_results.append(
            {
                "preprocessor": name,
                "rmse_test_mean": rmse.mean(),
                "rmse_test_std": rmse.std(),
                "rmse_train_mean": 0.0,
                "rmse_train_std": 0.0,
            }
        )
    xo_results_df = (
        pd.DataFrame(xo_results).set_index("preprocessor").sort_values("rmse_test_mean")
    )

    sk_fig = _build_results_plot(sk_results, "sklearn - Target Encoder Comparison")
    xo_fig = _build_results_plot(xo_results_df, "xorq - Target Encoder Comparison")

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(
        "Target Encoder Comparison: sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    out = "imgs/plot_target_encoder.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
