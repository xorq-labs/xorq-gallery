"""Model-based and sequential feature selection
==============================================

sklearn: Load diabetes dataset, apply SequentialFeatureSelector with forward and
backward directions using Ridge regression to select 2 best features, compare
selected features and cross-validation scores.

xorq: Same feature selection wrapped in Pipeline.from_instance, deferred fit to
extract selected features, deferred cross-validation scoring.

Both produce identical feature selections.

Dataset: load_diabetes (sklearn diabetes progression dataset)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from time import time
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
    df["target"] = y
    df["row_idx"] = range(len(df))

    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_feature_importance(
    forward_features, backward_features, forward_score, backward_score, title_prefix
):
    """Build feature selection comparison visualization.

    Parameters
    ----------
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
    ax.text(0.1, 0.65, forward_text, ha="left", va="top", fontsize=12,
            family="monospace", bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

    # Backward selection results
    backward_text = (
        f"Backward Selection:\n"
        f"  Selected features: {', '.join(backward_features)}\n"
        f"  CV Score: {backward_score:.4f}"
    )
    ax.text(0.1, 0.35, backward_text, ha="left", va="top", fontsize=12,
            family="monospace", bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))

    # Comparison
    match = "✓" if set(forward_features) == set(backward_features) else "✗"
    comparison_text = f"Both methods selected same features: {match}"
    ax.text(0.5, 0.05, comparison_text, ha="center", va="center", fontsize=12,
            fontweight="bold")

    plt.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit, extract selected features
# =========================================================================


def sklearn_way(df):
    """Eager sklearn: fit SequentialFeatureSelector with forward/backward, extract features.

    Returns
    -------
    dict
        Keys: "forward", "backward"
        Values: dict with "features", "score"
    """
    feature_names = [col for col in df.columns if col not in ["target", "row_idx"]]
    X = df[feature_names].values
    y = df["target"].values

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
    forward_features = [feature_names[i] for i in range(len(feature_names)) if forward_mask[i]]

    # Score with cross-validation
    forward_score = np.mean(
        cross_val_score(ridge, X[:, forward_mask], y, cv=CV_FOLDS)
    )

    print(f"    sklearn forward: selected {forward_features}, CV score = {forward_score:.4f}, time = {forward_time:.2f}s")
    results["forward"] = {
        "features": forward_features,
        "score": forward_score,
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
    backward_features = [feature_names[i] for i in range(len(feature_names)) if backward_mask[i]]

    # Score with cross-validation
    backward_score = np.mean(
        cross_val_score(ridge, X[:, backward_mask], y, cv=CV_FOLDS)
    )

    print(f"    sklearn backward: selected {backward_features}, CV score = {backward_score:.4f}, time = {backward_time:.2f}s")
    results["backward"] = {
        "features": backward_features,
        "score": backward_score,
        "mask": backward_mask,
    }

    return results


# =========================================================================
# XORQ WAY -- deferred fit, deferred feature extraction
# =========================================================================


def xorq_way(df):
    """Deferred xorq: wrap SequentialFeatureSelector in Pipeline.from_instance, fit deferred.

    Returns dict with fitted pipelines for extracting features in main().
    Nothing is executed until ``.execute()``.
    """
    con = xo.connect()

    # Register dataset
    feature_names = [col for col in df.columns if col not in ["target", "row_idx"]]
    table = con.register(df, "diabetes")

    # Feature columns
    features = tuple(feature_names)

    results = {}

    # Forward selection
    print("  Running forward selection (deferred)...")
    ridge_forward = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    sfs_forward = SequentialFeatureSelector(
        ridge_forward,
        n_features_to_select=N_FEATURES_TO_SELECT,
        direction="forward",
        cv=CV_FOLDS,
        n_jobs=-1,
    )
    sklearn_pipe_forward = SklearnPipeline([("sfs", sfs_forward)])
    xorq_pipe_forward = Pipeline.from_instance(sklearn_pipe_forward)

    # Fit deferred
    fitted_forward = xorq_pipe_forward.fit(table, features=features, target="target")

    results["forward"] = {
        "fitted": fitted_forward,
        "ridge": ridge_forward,
    }

    # Backward selection
    print("  Running backward selection (deferred)...")
    ridge_backward = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    sfs_backward = SequentialFeatureSelector(
        ridge_backward,
        n_features_to_select=N_FEATURES_TO_SELECT,
        direction="backward",
        cv=CV_FOLDS,
        n_jobs=-1,
    )
    sklearn_pipe_backward = SklearnPipeline([("sfs", sfs_backward)])
    xorq_pipe_backward = Pipeline.from_instance(sklearn_pipe_backward)

    # Fit deferred
    fitted_backward = xorq_pipe_backward.fit(table, features=features, target="target")

    results["backward"] = {
        "fitted": fitted_backward,
        "ridge": ridge_backward,
    }

    results["table"] = table
    results["feature_names"] = feature_names

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(df)

    print("\n=== XORQ WAY ===")
    xo_deferred = xorq_way(df)

    # Extract features from fitted pipelines
    feature_names = xo_deferred["feature_names"]
    y = df["target"].values

    # Forward selection
    fitted_forward = xo_deferred["forward"]["fitted"]
    sfs_model_forward = fitted_forward.fitted_steps[0].model
    forward_mask = sfs_model_forward.get_support()
    forward_features = [feature_names[i] for i in range(len(feature_names)) if forward_mask[i]]

    # Score with cross-validation
    X_forward = df[forward_features].values
    forward_score = np.mean(
        cross_val_score(xo_deferred["forward"]["ridge"], X_forward, y, cv=CV_FOLDS)
    )
    print(f"    xorq   forward: selected {forward_features}, CV score = {forward_score:.4f}")

    # Backward selection
    fitted_backward = xo_deferred["backward"]["fitted"]
    sfs_model_backward = fitted_backward.fitted_steps[0].model
    backward_mask = sfs_model_backward.get_support()
    backward_features = [feature_names[i] for i in range(len(feature_names)) if backward_mask[i]]

    # Score with cross-validation
    X_backward = df[backward_features].values
    backward_score = np.mean(
        cross_val_score(xo_deferred["backward"]["ridge"], X_backward, y, cv=CV_FOLDS)
    )
    print(f"    xorq   backward: selected {backward_features}, CV score = {backward_score:.4f}")

    xo_results = {
        "forward": {"features": forward_features, "score": forward_score, "mask": forward_mask},
        "backward": {"features": backward_features, "score": backward_score, "mask": backward_mask},
    }

    # Assert numerical equivalence BEFORE plotting
    print("\n=== ASSERTIONS ===")
    print("Comparing feature selections (sklearn vs xorq):")

    # Forward selection
    sk_forward_features = set(sk_results["forward"]["features"])
    xo_forward_features = set(xo_results["forward"]["features"])
    print(f"  sklearn forward: {sorted(sk_forward_features)}")
    print(f"  xorq   forward: {sorted(xo_forward_features)}")
    assert sk_forward_features == xo_forward_features, "Forward selection features don't match!"

    sk_forward_score = sk_results["forward"]["score"]
    xo_forward_score = xo_results["forward"]["score"]
    np.testing.assert_allclose(sk_forward_score, xo_forward_score, rtol=1e-3)
    print(f"  Forward CV scores match: {sk_forward_score:.4f}")

    # Backward selection
    sk_backward_features = set(sk_results["backward"]["features"])
    xo_backward_features = set(xo_results["backward"]["features"])
    print(f"  sklearn backward: {sorted(sk_backward_features)}")
    print(f"  xorq   backward: {sorted(xo_backward_features)}")
    assert sk_backward_features == xo_backward_features, "Backward selection features don't match!"

    sk_backward_score = sk_results["backward"]["score"]
    xo_backward_score = xo_results["backward"]["score"]
    np.testing.assert_allclose(sk_backward_score, xo_backward_score, rtol=1e-3)
    print(f"  Backward CV scores match: {sk_backward_score:.4f}")

    print("Assertions passed: sklearn and xorq feature selections match.")

    # Create deferred plot
    def _build_plot(df_unused):
        """Build feature selection plot from extracted results."""
        return _plot_feature_importance(
            forward_features=forward_features,
            backward_features=backward_features,
            forward_score=forward_score,
            backward_score=backward_score,
            title_prefix="xorq",
        )

    # Execute deferred plot
    print("\n=== PLOTTING ===")
    xo_png = deferred_matplotlib_plot(xo_deferred["table"], _build_plot).execute()

    # Build sklearn plot
    sk_fig = _plot_feature_importance(
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

    plt.suptitle("Sequential Feature Selection on Diabetes: sklearn vs xorq", fontsize=14)
    plt.tight_layout()
    out = "imgs/plot_select_from_model_diabetes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Composite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
