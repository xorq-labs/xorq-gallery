"""Visualizing Cross-Validation Behavior
=========================================

sklearn: Generate synthetic dataset with 100 samples, 3 classes, and 10 groups.
Visualize how 5 cross-validation strategies (KFold, StratifiedKFold, ShuffleSplit,
StratifiedShuffleSplit, TimeSeriesSplit) split the data into train/test sets
across 4 folds.

xorq: Same CV strategies executed via deferred_cross_val_score, which computes
fold assignments inside a UDWF (User-Defined Window Function). The fold_expr
attribute returns fold assignments (0=unused, 1=train, 2=test) computed entirely
within xorq's deferred execution engine.

sklearn's splitter is run on the same row-ordered data that xorq's UDWF saw,
so both sides produce identical fold assignments.

Dataset: Synthetic (100 samples, 3 classes, 10 groups)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/model_selection/plot_cv_indices.py
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.base import clone as _clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from xorq.expr.ml.cross_validation import deferred_cross_val_score
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    split_data_nop,
)
from xorq_gallery.utils import save_fig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 1338
N_SPLITS = 4
N_POINTS = 100

FEATURE_COLS = tuple(f"x_{i}" for i in range(10))
TARGET_COL = "y"
PRED_COL = "pred"  # unused, required by comparator

# Colors for fold bars
TRAIN_COLOR = "#3575D5"
TEST_COLOR = "#E8432A"
UNUSED_COLOR = "#E0E0E0"
CMAP_DATA = plt.cm.Paired

# Splitter definitions: (name, factory, order_by)
# Group-based splitters (GroupKFold, StratifiedGroupKFold, GroupShuffleSplit)
# are not supported by deferred_cross_val_score.
SPLITTERS = {
    "KFold": (
        lambda: KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        None,
    ),
    "StratifiedKFold": (
        lambda: StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        ),
        None,
    ),
    "ShuffleSplit": (
        lambda: ShuffleSplit(
            n_splits=N_SPLITS, test_size=0.25, random_state=RANDOM_STATE
        ),
        None,
    ),
    "StratifiedShuffleSplit": (
        lambda: StratifiedShuffleSplit(
            n_splits=N_SPLITS, test_size=0.25, random_state=RANDOM_STATE
        ),
        None,
    ),
    "TimeSeriesSplit": (
        lambda: TimeSeriesSplit(n_splits=N_SPLITS),
        "t",
    ),
}

# The sklearn pipeline used by deferred_cross_val_score (needed for fitting,
# but we only care about the fold assignments, not the scores).
_sk_pipeline = SklearnPipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    ]
)
_xorq_pipeline = Pipeline.from_instance(_sk_pipeline)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def load_data():
    """Generate synthetic dataset with classes and groups."""
    rng = np.random.RandomState(RANDOM_STATE)

    X = rng.randn(N_POINTS, 10)

    # Uneven class distribution
    percentiles_classes = [0.1, 0.3, 0.6]
    y = np.hstack(
        [[ii] * int(N_POINTS * perc) for ii, perc in enumerate(percentiles_classes)]
    )

    # Uneven group distribution
    group_prior = rng.dirichlet([2] * 10)
    groups = np.repeat(np.arange(10), rng.multinomial(N_POINTS, group_prior))

    df = pd.DataFrame(X, columns=list(FEATURE_COLS))
    df[TARGET_COL] = y
    df["group"] = groups
    df["t"] = range(len(df))
    return df


# ---------------------------------------------------------------------------
# sklearn fold assignment helper
# ---------------------------------------------------------------------------


def _build_sklearn_fold_df(cv, X_arr, y_arr):
    """Build fold DataFrame from sklearn splitter.

    Encoding: 0=unused, 1=train, 2=test.
    """

    def _make_col(train_idx, test_idx):
        col = np.zeros(len(y_arr), dtype=np.int8)
        col[train_idx] = 1
        col[test_idx] = 2
        return col

    return pd.DataFrame(
        {
            f"fold_{fold_i}": _make_col(train_idx, test_idx)
            for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_arr, y_arr))
        }
    )


# ---------------------------------------------------------------------------
# Comparator callbacks: make_sklearn_result / make_deferred_xorq_result
# ---------------------------------------------------------------------------

# Map pipeline id -> splitter name, so callbacks can look up which splitter
# to use. Each splitter gets its own clone of _sk_pipeline (see names_pipelines).
_pipeline_to_splitter = {}


def make_sklearn_cv_result(
    pipeline,
    train_data,
    test_data,
    features,
    target,
    metrics_names_funcs,
):
    """Sklearn side: run the splitter eagerly on the training data."""
    name = _pipeline_to_splitter[id(pipeline)]
    make_cv, _ = SPLITTERS[name]

    sklearn_fold_df = _build_sklearn_fold_df(
        make_cv(),
        train_data[list(features)].values,
        train_data[target].values,
    )
    return {
        "fitted": None,
        "preds": None,
        "metrics": {},
        "other": {"fold_df": sklearn_fold_df},
    }


def make_deferred_xorq_cv_result(
    pipeline,
    train_data,
    test_data,
    features,
    target,
    metrics_names_funcs,
    pred,
):
    """Xorq side: build deferred fold assignments via deferred_cross_val_score."""
    name = _pipeline_to_splitter[id(pipeline)]
    make_cv, order_by = SPLITTERS[name]

    result = deferred_cross_val_score(
        Pipeline.from_instance(pipeline),
        train_data,
        features=features,
        target=target,
        cv=make_cv(),
        random_seed=RANDOM_STATE,
        order_by=order_by,
    )
    return {
        "xorq_fitted": None,
        "preds": result.fold_expr,
        "metrics": {},
        "other": {},
    }


def make_xorq_cv_result(deferred_xorq_result):
    """Materialize the deferred fold_expr."""
    fold_expr = deferred_xorq_result["preds"]
    return {
        "fitted": None,
        "preds": None,
        "metrics": {},
        "other": {"fold_df": fold_expr.execute()},
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_fold_bars(ax, fold_values, n_splits, y_values, group_values, title):
    """Plot fold assignments as horizontal bars.

    fold_values : dict[str, np.ndarray]  — fold_i -> array of 0/1/2
    """
    n_samples = len(y_values)
    color_map = {0: UNUSED_COLOR, 1: TRAIN_COLOR, 2: TEST_COLOR}

    for fold_i in range(n_splits):
        indices = fold_values[f"fold_{fold_i}"]
        colors = [color_map[v] for v in indices]
        ax.barh(
            [fold_i] * n_samples,
            width=1,
            left=range(n_samples),
            height=0.8,
            color=colors,
            edgecolor="none",
        )

    # Class labels bar
    n_classes = max(len(set(y_values)), 1)
    class_colors = [CMAP_DATA(c / max(n_classes - 1, 1)) for c in y_values]
    ax.barh(
        [n_splits] * n_samples,
        width=1,
        left=range(n_samples),
        height=0.8,
        color=class_colors,
        edgecolor="none",
    )

    # Group labels bar
    n_groups = max(len(set(group_values)), 1)
    group_colors = [CMAP_DATA(g / max(n_groups - 1, 1)) for g in group_values]
    ax.barh(
        [n_splits + 1] * n_samples,
        width=1,
        left=range(n_samples),
        height=0.8,
        color=group_colors,
        edgecolor="none",
    )

    ax.set(
        yticks=range(n_splits + 2),
        yticklabels=[f"Fold {i}" for i in range(n_splits)] + ["Class", "Group"],
        xlabel="Sample index",
        xlim=(0, n_samples),
        ylim=(n_splits + 1.8, -0.5),
    )
    ax.set_title(title, fontsize=12, fontweight="bold")


# ---------------------------------------------------------------------------
# Comparator callbacks: compare_results / plot_results
# ---------------------------------------------------------------------------


def compare_results(comparator):
    """Assert that sklearn and xorq fold assignments match for every splitter."""
    print("\n=== Comparing Results ===")
    for name in SPLITTERS:
        xo_fold_df = comparator.xorq_results[name]["other"]["fold_df"]

        # sklearn ran on raw data order; xorq ran on deterministically-sorted
        # data. To compare, re-run sklearn on the xorq row order.
        make_cv, _ = SPLITTERS[name]
        xo_sklearn_fold_df = _build_sklearn_fold_df(
            make_cv(),
            xo_fold_df[list(FEATURE_COLS)].values,
            xo_fold_df[TARGET_COL].values,
        )

        for fold_i in range(N_SPLITS):
            col = f"fold_{fold_i}"
            np.testing.assert_array_equal(
                xo_sklearn_fold_df[col].values,
                xo_fold_df[col].values,
                err_msg=f"{name} fold {fold_i} mismatch",
            )
        print(f"  {name}: fold assignments match")

    print("  All fold assignments verified.")


def plot_results(comparator):
    """Build side-by-side fold assignment visualization."""
    n_splitters = len(SPLITTERS)
    fig, axes = plt.subplots(
        n_splitters, 2, figsize=(16, 2.8 * n_splitters), constrained_layout=True
    )

    for row, name in enumerate(SPLITTERS):
        sk_fold_df = comparator.sklearn_results[name]["other"]["fold_df"]
        xo_fold_df = comparator.xorq_results[name]["other"]["fold_df"]

        # Use xorq's materialized data for class/group display
        y_display = xo_fold_df[TARGET_COL].values
        group_display = xo_fold_df["group"].values

        sk_folds = {
            f"fold_{i}": sk_fold_df[f"fold_{i}"].values for i in range(N_SPLITS)
        }
        xo_folds = {
            f"fold_{i}": xo_fold_df[f"fold_{i}"].values for i in range(N_SPLITS)
        }

        _plot_fold_bars(
            axes[row, 0],
            sk_folds,
            N_SPLITS,
            y_display,
            group_display,
            f"{name} (sklearn)",
        )
        _plot_fold_bars(
            axes[row, 1],
            xo_folds,
            N_SPLITS,
            y_display,
            group_display,
            f"{name} (xorq)",
        )

    # Legend
    legend_patches = [
        Patch(color=TRAIN_COLOR, label="Train"),
        Patch(color=TEST_COLOR, label="Test"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=11)
    fig.suptitle(
        "CV Fold Assignments: sklearn vs xorq (deferred)",
        fontsize=16,
        fontweight="bold",
    )
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

metrics_names_funcs = ()

# Each splitter gets its own pipeline clone so we can map pipeline id -> name.
names_pipelines = tuple((name, _clone(_sk_pipeline)) for name in SPLITTERS)
for name, pipeline in names_pipelines:
    _pipeline_to_splitter[id(pipeline)] = name

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=TARGET_COL,
    pred=PRED_COL,
    metrics_names_funcs=metrics_names_funcs,
    load_data=load_data,
    split_data=split_data_nop,
    make_sklearn_result=make_sklearn_cv_result,
    make_deferred_xorq_result=make_deferred_xorq_cv_result,
    make_xorq_result=make_xorq_cv_result,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)

# Expose deferred fold exprs for xorq build
xorq_kfold_folds = comparator.deferred_xorq_results["KFold"]["preds"]
xorq_stratifiedkfold_folds = comparator.deferred_xorq_results["StratifiedKFold"][
    "preds"
]
xorq_shufflesplit_folds = comparator.deferred_xorq_results["ShuffleSplit"]["preds"]
xorq_stratifiedshufflesplit_folds = comparator.deferred_xorq_results[
    "StratifiedShuffleSplit"
]["preds"]
xorq_timeseriessplit_folds = comparator.deferred_xorq_results["TimeSeriesSplit"][
    "preds"
]


# =========================================================================
# Main
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    comparator.result_comparison
    save_fig("imgs/plot_cv_indices_all.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
