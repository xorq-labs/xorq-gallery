"""Model-based and sequential feature selection
==============================================

sklearn: Load diabetes dataset, apply SequentialFeatureSelector with forward and
backward directions using Ridge regression to select 2 best features, compare
selected features and cross-validation scores via cross_val_score.

xorq: Same feature selection wrapped in Pipeline.from_instance, deferred fit to
extract selected features, deferred cross-validation scoring via
deferred_cross_val_score with KFold.

Both produce identical feature selections and per-fold CV scores.

Dataset: load_diabetes (sklearn diabetes progression dataset)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/feature_selection/plot_select_from_model_diabetes.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from xorq.expr.ml.cross_validation import deferred_cross_val_score
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    split_data_nop,
)
from xorq_gallery.utils import fig_to_image, save_fig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FEATURES_TO_SELECT = 2
RANDOM_STATE = 0
CV_FOLDS = 3
TARGET_COL = "target"
ROW_IDX = "row_idx"

CV_SPLITTER = KFold(n_splits=CV_FOLDS, shuffle=False)

methods = (FORWARD, BACKWARD) = ("forward", "backward")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data():
    """Load the diabetes dataset and return as pandas DataFrame with feature names."""
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df[TARGET_COL] = diabetes.target
    df[ROW_IDX] = range(len(df))
    _load_data.feature_names = list(diabetes.feature_names)
    return df.sort_values(ROW_IDX).reset_index(drop=True)


# Force data load to set metadata
_load_data()
FEATURE_COLS = tuple(_load_data.feature_names)


# ---------------------------------------------------------------------------
# Pipelines — one per direction
# ---------------------------------------------------------------------------


def _build_sfs_pipeline(direction):
    """Build sklearn Pipeline containing SequentialFeatureSelector."""
    return SklearnPipeline(
        [
            (
                "sfs",
                SequentialFeatureSelector(
                    Ridge(alpha=1.0, random_state=RANDOM_STATE),
                    n_features_to_select=N_FEATURES_TO_SELECT,
                    direction=direction,
                    cv=CV_FOLDS,
                    n_jobs=-1,
                ),
            )
        ]
    )


names_pipelines = (
    (FORWARD, _build_sfs_pipeline("forward")),
    (BACKWARD, _build_sfs_pipeline("backward")),
)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_feature_selection(results, title):
    """Text-based feature selection results visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.text(0.5, 0.9, title, ha="center", va="center", fontsize=16, fontweight="bold")

    for i, (direction, text_y) in enumerate(zip(methods, (0.65, 0.35))):
        features = results[direction]["other"]["selected_features"]
        score = results[direction]["other"]["cv_score"]
        text = (
            f"{direction.title()} Selection:\n"
            f"  Selected features: {', '.join(features)}\n"
            f"  CV Score: {score:.4f}"
        )
        color = "lightblue" if direction == "forward" else "lightgreen"
        ax.text(
            0.1, text_y, text,
            ha="left", va="top", fontsize=12, family="monospace",
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.5),
        )

    fwd_features = set(results[FORWARD]["other"]["selected_features"])
    bwd_features = set(results[BACKWARD]["other"]["selected_features"])
    match = "Yes" if fwd_features == bwd_features else "No"
    ax.text(
        0.5, 0.05,
        f"Both methods selected same features: {match}",
        ha="center", va="center", fontsize=12, fontweight="bold",
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Custom make_*_result — SFS fit + Ridge CV scoring
# ---------------------------------------------------------------------------


def _make_sklearn_result(
    pipeline, train_data, test_data, features, target, metrics_names_funcs
):
    """Sklearn: fit SFS, extract selected features, cross-validate Ridge on them."""
    feature_names = list(features)
    X = train_data[feature_names].values
    y = train_data[target].values

    fitted = clone(pipeline).fit(X, y)
    sfs_model = fitted.steps[0][1]
    mask = sfs_model.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if mask[i]]

    # Score Ridge on selected features
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    scores = cross_val_score(ridge, X[:, mask], y, cv=CV_SPLITTER)
    mean_score = np.mean(scores)

    return {
        "fitted": sfs_model,
        "preds": scores,
        "metrics": {"cv_score_mean": mean_score},
        "other": {
            "selected_features": selected_features,
            "cv_score": mean_score,
            "cv_scores": scores,
            "mask": mask,
        },
    }


def _make_deferred_xorq_result(
    pipeline, train_data, test_data, features, target, metrics_names_funcs, pred
):
    """Deferred xorq: fit SFS, extract selected features, deferred_cross_val_score."""
    feature_names = list(features)
    xorq_pipe = Pipeline.from_instance(pipeline)
    fitted = xorq_pipe.fit(train_data, features=features, target=target)

    sfs_model = fitted.fitted_steps[0].model
    mask = sfs_model.get_support()
    selected_features = tuple(
        feature_names[i] for i in range(len(feature_names)) if mask[i]
    )

    # Score Ridge on selected features via deferred CV
    ridge_pipe = Pipeline.from_instance(
        SklearnPipeline([("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE))])
    )
    cv_result = deferred_cross_val_score(
        ridge_pipe, train_data, selected_features, target,
        cv=CV_SPLITTER, order_by=ROW_IDX,
    )

    return {
        "xorq_fitted": fitted,
        "preds": cv_result,
        "metrics": {},
        "other": {
            "selected_features": lambda: list(selected_features),
            "mask": lambda: mask,
        },
    }


def _make_xorq_result(deferred_xorq_result):
    """Materialize deferred CV scores and feature info."""
    scores = deferred_xorq_result["preds"].execute()
    mean_score = scores.mean()
    other_raw = deferred_xorq_result.get("other", {})
    selected_features = other_raw["selected_features"]()
    mask = other_raw["mask"]()

    return {
        "fitted": deferred_xorq_result["xorq_fitted"].fitted_steps[0].model,
        "preds": scores,
        "metrics": {"cv_score_mean": mean_score},
        "other": {
            "selected_features": selected_features,
            "cv_score": mean_score,
            "cv_scores": scores,
            "mask": mask,
        },
    }


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def compare_results(comparator):
    assert sorted(sklearn_results := comparator.sklearn_results) == sorted(
        xorq_results := comparator.xorq_results
    )
    print("\n=== Comparing Results ===")
    for direction in methods:
        sk_features = set(sklearn_results[direction]["other"]["selected_features"])
        xo_features = set(xorq_results[direction]["other"]["selected_features"])
        assert sk_features == xo_features, f"{direction} features don't match!"
        print(f"  {direction}: features match {sorted(sk_features)}")

    sk_scores_df = pd.DataFrame(
        {d: comparator.sklearn_results[d]["other"]["cv_scores"] for d in methods}
    )
    xo_scores_df = pd.DataFrame(
        {d: comparator.xorq_results[d]["other"]["cv_scores"] for d in methods}
    )
    pd.testing.assert_frame_equal(sk_scores_df, xo_scores_df, rtol=1e-3)
    print("  Per-fold CV scores match for all directions.")
    print("Assertions passed.")


def plot_results(comparator):
    sk_fig = _plot_feature_selection(comparator.sklearn_results, "sklearn - Feature Selection")
    xo_fig = _plot_feature_selection(comparator.xorq_results, "xorq - Feature Selection")

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn")
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq")
    axes[1].axis("off")

    fig.suptitle(
        "Sequential Feature Selection on Diabetes: sklearn vs xorq", fontsize=14
    )
    fig.tight_layout()

    plt.close(sk_fig)
    plt.close(xo_fig)
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=TARGET_COL,
    pred="pred",
    metrics_names_funcs=(),
    load_data=_load_data,
    split_data=split_data_nop,
    make_sklearn_result=_make_sklearn_result,
    make_deferred_xorq_result=_make_deferred_xorq_result,
    make_xorq_result=_make_xorq_result,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)

# Module-level deferred exprs
(xorq_forward_cv, xorq_backward_cv) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    comparator.result_comparison
    save_fig("imgs/plot_select_from_model_diabetes.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
