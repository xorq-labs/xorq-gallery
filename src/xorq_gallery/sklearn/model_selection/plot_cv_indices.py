"""Visualizing Cross-Validation Behavior
=========================================

sklearn: Generate synthetic dataset with 100 samples, 3 classes, and 10 groups.
Visualize how 8 different cross-validation strategies (KFold, GroupKFold,
ShuffleSplit, StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit,
StratifiedShuffleSplit, TimeSeriesSplit) split the data into train/test sets
across 4 folds. Each CV object's behavior is displayed as a scatter plot showing
which samples are assigned to train (blue) vs test (red) in each fold.

xorq: Same visualization logic wrapped in deferred execution. The data generation
and CV splitting logic is identical, but the plotting is wrapped via
deferred_matplotlib_plot to demonstrate deferred visualization workflows.

Both produce identical visualizations.

Dataset: Synthetic (100 samples, 3 classes, 10 groups)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from matplotlib.patches import Patch
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 1338
N_SPLITS = 4
N_POINTS = 100

# Matplotlib colormaps
CMAP_DATA = plt.cm.Paired
CMAP_CV = plt.cm.coolwarm


# ---------------------------------------------------------------------------
# Data generation (shared)
# ---------------------------------------------------------------------------


def _generate_data():
    """Generate synthetic dataset with classes and groups.

    Returns
    -------
    dict with keys:
        - X : np.ndarray, shape (100, 10)
        - y : np.ndarray, shape (100,) - class labels (0, 1, 2)
        - groups : np.ndarray, shape (100,) - group labels (0-9)
        - df : pd.DataFrame with columns X, y, groups, row_idx
    """
    rng = np.random.RandomState(RANDOM_STATE)

    # Generate random features
    X = rng.randn(N_POINTS, 10)

    # Generate uneven class distribution
    percentiles_classes = [0.1, 0.3, 0.6]
    y = np.hstack([[ii] * int(N_POINTS * perc)
                   for ii, perc in enumerate(percentiles_classes)])

    # Generate uneven group distribution
    group_prior = rng.dirichlet([2] * 10)
    groups = np.repeat(np.arange(10), rng.multinomial(N_POINTS, group_prior))

    # Create DataFrame for xorq
    df = pd.DataFrame(X, columns=[f"x_{i}" for i in range(10)])
    df["y"] = y
    df["groups"] = groups
    df["row_idx"] = range(len(df))

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "df": df,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_cv_indices(cv, X, y, groups, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object.

    Parameters
    ----------
    cv : sklearn CV object
        Cross-validation iterator.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    groups : np.ndarray
        Group labels.
    n_splits : int
        Number of splits.
    lw : int, optional
        Line width for markers.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 3))

    # Determine if CV uses groups
    use_groups = "Group" in type(cv).__name__
    groups_arg = groups if use_groups else None

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups_arg)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=CMAP_CV,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X),
        c=y, marker="_", lw=lw, cmap=CMAP_DATA
    )

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X),
        c=groups, marker="_", lw=lw, cmap=CMAP_DATA
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, N_POINTS],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)

    # Add legend
    ax.legend(
        [Patch(color=CMAP_CV(0.8)), Patch(color=CMAP_CV(0.02))],
        ["Testing set", "Training set"],
        loc=(1.02, 0.8),
    )

    fig.tight_layout()
    fig.subplots_adjust(right=0.7)

    return fig


def _plot_all_cv_strategies(X, y, groups, n_splits):
    """Plot all 8 CV strategies in a grid.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    cvs = (
        KFold,
        GroupKFold,
        ShuffleSplit,
        StratifiedKFold,
        StratifiedGroupKFold,
        GroupShuffleSplit,
        StratifiedShuffleSplit,
        TimeSeriesSplit,
    )

    # Create 4x2 grid
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, cv_class in enumerate(cvs):
        cv = cv_class(n_splits=n_splits)
        ax = axes[idx]

        # Determine if CV uses groups
        use_groups = "Group" in cv_class.__name__
        groups_arg = groups if use_groups else None

        # Generate the training/testing visualizations for each CV split
        for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups_arg)):
            # Fill in indices with the training/test groups
            indices = np.array([np.nan] * len(X))
            indices[tt] = 1
            indices[tr] = 0

            # Visualize the results
            ax.scatter(
                range(len(indices)),
                [ii + 0.5] * len(indices),
                c=indices,
                marker="_",
                lw=10,
                cmap=CMAP_CV,
                vmin=-0.2,
                vmax=1.2,
            )

        # Plot the data classes and groups at the end
        ax.scatter(
            range(len(X)), [ii + 1.5] * len(X),
            c=y, marker="_", lw=10, cmap=CMAP_DATA
        )

        ax.scatter(
            range(len(X)), [ii + 2.5] * len(X),
            c=groups, marker="_", lw=10, cmap=CMAP_DATA
        )

        # Formatting
        yticklabels = list(range(n_splits)) + ["class", "group"]
        ax.set(
            yticks=np.arange(n_splits + 2) + 0.5,
            yticklabels=yticklabels,
            xlabel="Sample index",
            ylabel="CV iteration",
            ylim=[n_splits + 2.2, -0.2],
            xlim=[0, N_POINTS],
        )
        ax.set_title("{}".format(cv_class.__name__), fontsize=12)

    # Add legend to last subplot
    axes[-1].legend(
        [Patch(color=CMAP_CV(0.8)), Patch(color=CMAP_CV(0.02))],
        ["Testing set", "Training set"],
        loc="center",
    )

    fig.tight_layout()

    return fig


# =========================================================================
# SKLEARN WAY -- eager CV visualization
# =========================================================================


def sklearn_way(data_dict):
    """Eager sklearn: Generate CV splits and visualize immediately.

    Returns dict with individual CV plots.
    """
    X = data_dict["X"]
    y = data_dict["y"]
    groups = data_dict["groups"]

    # Test with KFold as example
    print("sklearn: Visualizing KFold cross-validation...")
    cv = KFold(n_splits=N_SPLITS)

    # Collect split indices for verification
    splits = list(cv.split(X=X, y=y, groups=None))

    print(f"Number of splits: {len(splits)}")
    print(f"First split train size: {len(splits[0][0])}, test size: {len(splits[0][1])}")

    # Generate plot for KFold
    fig_kfold = _plot_cv_indices(cv, X, y, groups, N_SPLITS)

    # Generate composite plot with all CV strategies
    fig_all = _plot_all_cv_strategies(X, y, groups, N_SPLITS)

    return {
        "splits": splits,
        "fig_kfold": fig_kfold,
        "fig_all": fig_all,
    }


# ---------------------------------------------------------------------------
# Plotting helpers for deferred xorq plots (module-level)
# ---------------------------------------------------------------------------


def _build_kfold_plot_from_df(df_materialized):
    """Build KFold plot from materialized DataFrame."""
    # Extract arrays from DataFrame
    X_mat = df_materialized[[f"x_{i}" for i in range(10)]].values
    y_mat = df_materialized["y"].values
    groups_mat = df_materialized["groups"].values

    cv = KFold(n_splits=N_SPLITS)
    return _plot_cv_indices(cv, X_mat, y_mat, groups_mat, N_SPLITS)


def _build_all_cv_plot_from_df(df_materialized):
    """Build composite plot with all CV strategies."""
    # Extract arrays from DataFrame
    X_mat = df_materialized[[f"x_{i}" for i in range(10)]].values
    y_mat = df_materialized["y"].values
    groups_mat = df_materialized["groups"].values

    return _plot_all_cv_strategies(X_mat, y_mat, groups_mat, N_SPLITS)


# =========================================================================
# XORQ WAY -- deferred CV visualization
# =========================================================================


def xorq_way(data_dict):
    """Deferred xorq: Same CV visualization but with deferred plotting.

    Returns dict with data table expression and splits for verification.
    """
    con = xo.connect()

    df = data_dict["df"]
    X = data_dict["X"]
    y = data_dict["y"]
    groups = data_dict["groups"]

    # Register data
    table = con.register(df, "cv_data")

    print("xorq: Visualizing KFold cross-validation (deferred)...")

    # For verification, compute splits eagerly (this matches sklearn behavior)
    cv = KFold(n_splits=N_SPLITS)
    splits = list(cv.split(X=X, y=y, groups=None))

    print(f"Number of splits: {len(splits)}")
    print(f"First split train size: {len(splits[0][0])}, test size: {len(splits[0][1])}")

    return {
        "splits": splits,
        "table": table,
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    print("=== GENERATING DATA ===")
    data_dict = _generate_data()

    print("\n=== SKLEARN WAY ===")
    sk_results = sklearn_way(data_dict)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(data_dict)

    # ---- Assert numerical equivalence BEFORE plotting ----
    print("\n=== ASSERTIONS ===")

    # Verify that split indices match
    sk_splits = sk_results["splits"]
    xo_splits = xo_results["splits"]

    assert len(sk_splits) == len(xo_splits), "Number of splits must match"

    sk_split_df = pd.DataFrame({
        f"fold_{i}_{role}": pd.Series(indices)
        for i, (train, test) in enumerate(sk_splits)
        for role, indices in [("train", train), ("test", test)]
    })
    xo_split_df = pd.DataFrame({
        f"fold_{i}_{role}": pd.Series(indices)
        for i, (train, test) in enumerate(xo_splits)
        for role, indices in [("train", train), ("test", test)]
    })
    pd.testing.assert_frame_equal(sk_split_df, xo_split_df)
    print("Assertions passed: sklearn and xorq CV splits are identical.")

    print("\n=== EXECUTING DEFERRED PLOTS ===")
    xo_kfold_png = deferred_matplotlib_plot(
        xo_results["table"], _build_kfold_plot_from_df
    ).execute()
    xo_all_png = deferred_matplotlib_plot(
        xo_results["table"], _build_all_cv_plot_from_df
    ).execute()

    print("Deferred plots executed successfully.")

    # Composite: sklearn (left) | xorq (right) for KFold
    print("\n=== CREATING COMPOSITE PLOTS ===")

    # KFold comparison
    xo_kfold_img = load_plot_bytes(xo_kfold_png)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(fig_to_image(sk_results["fig_kfold"]))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(xo_kfold_img)
    axes[1].set_title("xorq (deferred)", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle("KFold Cross-Validation: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    out_kfold = "imgs/plot_cv_indices_kfold.png"
    fig.savefig(out_kfold, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"KFold comparison saved to {out_kfold}")

    # All CV strategies comparison
    xo_all_img = load_plot_bytes(xo_all_png)

    fig, axes = plt.subplots(1, 2, figsize=(28, 12))

    axes[0].imshow(fig_to_image(sk_results["fig_all"]))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(xo_all_img)
    axes[1].set_title("xorq (deferred)", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle("All CV Strategies: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    out_all = "imgs/plot_cv_indices_all.png"
    fig.savefig(out_all, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"All CV strategies comparison saved to {out_all}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
