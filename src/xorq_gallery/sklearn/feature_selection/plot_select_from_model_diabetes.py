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
"""

from __future__ import annotations

import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from toolz import curry
from xorq.expr.ml.cross_validation import deferred_cross_val_score
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FEATURES_TO_SELECT = 2
RANDOM_STATE = 0
CV_FOLDS = 3
TARGET_COL = "target"
ROW_IDX = "row_idx"

CV_SPLITTER = KFold(n_splits=CV_FOLDS, shuffle=False)


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Load the diabetes dataset and return as pandas DataFrame with feature names."""
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    feature_names = diabetes.feature_names

    # Build DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df[TARGET_COL] = y
    df[ROW_IDX] = range(len(df))

    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


@curry
def _plot_feature_importance(
    df_unused,
    forward_features,
    backward_features,
    forward_score,
    backward_score,
    title_prefix,
):
    """Build feature selection comparison visualization.

    Parameters
    ----------
    df_unused : DataFrame
        Unused DataFrame parameter (for deferred plotting compatibility)
    forward_features : list[str]
        Features selected by forward selection
    backward_features : list[str]
        Features selected by backward selection
    forward_score : float
        Cross-validation score for forward selection
    backward_score : float
        Cross-validation score for backward selection
    title_prefix : str
        Prefix for the title (e.g., "sklearn" or "xorq")

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a text-based visualization
    ax.axis("off")

    title = f"{title_prefix} - Feature Selection Results"
    ax.text(0.5, 0.9, title, ha="center", va="center", fontsize=16, fontweight="bold")

    # Forward selection results
    forward_text = (
        f"Forward Selection:\n"
        f"  Selected features: {', '.join(forward_features)}\n"
        f"  CV Score: {forward_score:.4f}"
    )
    ax.text(
        0.1,
        0.65,
        forward_text,
        ha="left",
        va="top",
        fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    # Backward selection results
    backward_text = (
        f"Backward Selection:\n"
        f"  Selected features: {', '.join(backward_features)}\n"
        f"  CV Score: {backward_score:.4f}"
    )
    ax.text(
        0.1,
        0.35,
        backward_text,
        ha="left",
        va="top",
        fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

    # Comparison
    match = "✓" if set(forward_features) == set(backward_features) else "✗"
    comparison_text = f"Both methods selected same features: {match}"
    ax.text(
        0.5,
        0.05,
        comparison_text,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit, extract selected features
# =========================================================================


def sklearn_way(df):
    """Eager sklearn: fit SequentialFeatureSelector with forward/backward, extract features.

    df must be sorted by ROW_IDX so KFold fold assignments match the xorq side.

    Returns
    -------
    dict
        Keys: "forward", "backward"
        Values: dict with "features", "score", "scores"
    """
    feature_names = [col for col in df.columns if col not in (TARGET_COL, ROW_IDX)]
    X = df[feature_names].values
    y = df[TARGET_COL].values

    results = {}

    # Forward selection
    print("  Running forward selection...")
    t0 = time()
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    sfs_forward = SequentialFeatureSelector(
        ridge,
        n_features_to_select=N_FEATURES_TO_SELECT,
        direction="forward",
        cv=CV_FOLDS,
        n_jobs=-1,
    )
    sfs_forward.fit(X, y)
    forward_time = time() - t0

    # Get selected feature indices and names
    forward_mask = sfs_forward.get_support()
    forward_features = [
        feature_names[i] for i in range(len(feature_names)) if forward_mask[i]
    ]

    # Score with cross-validation
    forward_scores = cross_val_score(ridge, X[:, forward_mask], y, cv=CV_SPLITTER)
    forward_score = np.mean(forward_scores)

    print(
        f"    sklearn forward: selected {forward_features}, CV score = {forward_score:.4f}, time = {forward_time:.2f}s"
    )
    results["forward"] = {
        "features": forward_features,
        "score": forward_score,
        "scores": forward_scores,
        "mask": forward_mask,
    }

    # Backward selection
    print("  Running backward selection...")
    t0 = time()
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    sfs_backward = SequentialFeatureSelector(
        ridge,
        n_features_to_select=N_FEATURES_TO_SELECT,
        direction="backward",
        cv=CV_FOLDS,
        n_jobs=-1,
    )
    sfs_backward.fit(X, y)
    backward_time = time() - t0

    # Get selected feature indices and names
    backward_mask = sfs_backward.get_support()
    backward_features = [
        feature_names[i] for i in range(len(feature_names)) if backward_mask[i]
    ]

    # Score with cross-validation
    backward_scores = cross_val_score(ridge, X[:, backward_mask], y, cv=CV_SPLITTER)
    backward_score = np.mean(backward_scores)

    print(
        f"    sklearn backward: selected {backward_features}, CV score = {backward_score:.4f}, time = {backward_time:.2f}s"
    )
    results["backward"] = {
        "features": backward_features,
        "score": backward_score,
        "scores": backward_scores,
        "mask": backward_mask,
    }

    return results


# =========================================================================
# XORQ WAY -- deferred fit, deferred feature extraction
# =========================================================================


def xorq_way(df):
    """Deferred xorq: wrap SequentialFeatureSelector in Pipeline.from_instance, fit deferred.

    After extracting selected features, score the Ridge model on those features
    using deferred_cross_val_score. Nothing is executed until ``.execute()``.
    """
    con = xo.connect()

    # Register dataset
    feature_names = [col for col in df.columns if col not in (TARGET_COL, ROW_IDX)]
    table = con.register(df, "diabetes")

    # Feature columns
    features = tuple(feature_names)

    results = {}

    for direction in ("forward", "backward"):
        print(f"  Running {direction} selection (deferred)...")
        ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        sfs = SequentialFeatureSelector(
            ridge,
            n_features_to_select=N_FEATURES_TO_SELECT,
            direction=direction,
            cv=CV_FOLDS,
            n_jobs=-1,
        )
        sklearn_pipe = SklearnPipeline([("sfs", sfs)])
        xorq_pipe = Pipeline.from_instance(sklearn_pipe)

        # Fit deferred to get selected features
        fitted = xorq_pipe.fit(table, features=features, target=TARGET_COL)

        # Extract selected feature names from the fitted model
        sfs_model = fitted.fitted_steps[0].model
        mask = sfs_model.get_support()
        selected_features = tuple(
            feature_names[i] for i in range(len(feature_names)) if mask[i]
        )
        print(f"    xorq {direction}: selected {list(selected_features)}")

        # Score Ridge on selected features via deferred_cross_val_score
        ridge_pipe = Pipeline.from_instance(
            SklearnPipeline([("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE))])
        )
        cv_result = deferred_cross_val_score(
            ridge_pipe,
            table,
            selected_features,
            TARGET_COL,
            cv=CV_SPLITTER,
            order_by=ROW_IDX,
        )

        results[direction] = {
            "features": list(selected_features),
            "mask": mask,
            "cv_result": cv_result,
        }

    results["table"] = table

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()
    # Sort by ROW_IDX so sklearn KFold sees the same row order as xorq
    df = df.sort_values(ROW_IDX).reset_index(drop=True)

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(df)

    print("\n=== XORQ WAY ===")
    xo_deferred = xorq_way(df)

    # Execute deferred CV scores
    xo_results = {}
    for direction in ("forward", "backward"):
        scores = xo_deferred[direction]["cv_result"].execute()
        mean_score = scores.mean()
        features = xo_deferred[direction]["features"]
        print(f"    xorq {direction}: CV score = {mean_score:.4f}")
        xo_results[direction] = {
            "features": features,
            "score": mean_score,
            "scores": scores,
            "mask": xo_deferred[direction]["mask"],
        }

    # Assert numerical equivalence BEFORE plotting
    print("\n=== ASSERTIONS ===")
    for direction in ("forward", "backward"):
        sk_features = set(sk_results[direction]["features"])
        xo_features = set(xo_results[direction]["features"])
        assert sk_features == xo_features, (
            f"{direction} selection features don't match!"
        )
        print(f"  {direction}: features match {sorted(sk_features)}")

    sk_scores_df = pd.DataFrame(
        {d: sk_results[d]["scores"] for d in ("forward", "backward")}
    )
    xo_scores_df = pd.DataFrame(
        {d: xo_results[d]["scores"] for d in ("forward", "backward")}
    )
    pd.testing.assert_frame_equal(sk_scores_df, xo_scores_df, rtol=1e-3)
    print("  Per-fold CV scores match for all directions.")
    print("Assertions passed.")

    # Execute deferred plot
    print("\n=== PLOTTING ===")
    xo_png = deferred_matplotlib_plot(
        xo_deferred["table"],
        _plot_feature_importance(
            forward_features=tuple(xo_results["forward"]["features"]),
            backward_features=tuple(xo_results["backward"]["features"]),
            forward_score=xo_results["forward"]["score"],
            backward_score=xo_results["backward"]["score"],
            title_prefix="xorq",
        ),
    ).execute()

    # Build sklearn plot
    sk_fig = _plot_feature_importance(
        df,
        forward_features=sk_results["forward"]["features"],
        backward_features=sk_results["backward"]["features"],
        forward_score=sk_results["forward"]["score"],
        backward_score=sk_results["backward"]["score"],
        title_prefix="sklearn",
    )

    # Composite: sklearn (left) | xorq (right)
    xo_img = load_plot_bytes(xo_png)
    sk_img = fig_to_image(sk_fig)

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes[0].imshow(sk_img)
    axes[0].set_title("sklearn")
    axes[0].axis("off")

    axes[1].imshow(xo_img)
    axes[1].set_title("xorq")
    axes[1].axis("off")

    fig.suptitle(
        "Sequential Feature Selection on Diabetes: sklearn vs xorq", fontsize=14
    )
    fig.tight_layout()
    out = "imgs/plot_select_from_model_diabetes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Composite plot saved to {out}")


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
