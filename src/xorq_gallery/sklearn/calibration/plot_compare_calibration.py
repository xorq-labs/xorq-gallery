"""Comparison of Calibration of Classifiers
========================================

sklearn: Compare calibration of four different classifiers (Logistic Regression,
Naive Bayes, Linear SVC, Random Forest) on binary classification with 100,000
samples (100 train, 99,900 test). Uses CalibrationDisplay to plot calibration
curves and histograms showing probability distributions.

xorq: Same classifiers wrapped in Pipeline.from_instance, deferred fit and
predict_proba, deferred calibration curves and histograms via
deferred_matplotlib_plot.

Both produce identical calibration metrics proving model reliability.

Dataset: make_classification (sklearn synthetic, 100k samples, 20 features)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.svm import LinearSVC
from toolz import curry
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_SAMPLES = 100_000
N_FEATURES = 20
N_INFORMATIVE = 2
N_REDUNDANT = 2
TRAIN_SAMPLES = 100  # Very small training set
N_BINS = 10

TARGET_COL = "y"
ROW_IDX = "row_idx"
CLF_NAME_COL = "clf_name"
Y_TRUE_COL = "y_true"
Y_PROB_COL = "y_prob"


# ---------------------------------------------------------------------------
# Custom SVC with predict_proba (matching sklearn example)
# ---------------------------------------------------------------------------


class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with predict_proba method that naively scales
    decision_function output."""

    def fit(self, X, y):
        super().fit(X, y)
        df = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()
        return self

    def predict_proba(self, X):
        """Min-max scale output of decision_function to [0,1]."""
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Generate synthetic binary classification dataset.

    Matches sklearn example: 100k samples, 20 features (2 informative, 2 redundant).
    Uses shuffle=False split to get stable calibration estimates.
    """
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        random_state=RANDOM_STATE,
    )

    # Create dataframe
    feature_cols = tuple(f"x{i}" for i in range(N_FEATURES))
    df = pd.DataFrame(X, columns=feature_cols)
    df[TARGET_COL] = y
    df[ROW_IDX] = range(len(df))

    return df, feature_cols


def _build_classifiers():
    """Return tuple of (clf, name) tuples matching sklearn example.

    Includes:
    - Logistic Regression with CV (auto-tuned regularization)
    - Gaussian Naive Bayes
    - Linear SVC with naive probability calibration
    - Random Forest
    """
    lr = LogisticRegressionCV(
        Cs=tuple(np.logspace(-6, 6, 101).tolist()),
        cv=10,
        l1_ratios=(0,),
        scoring="neg_log_loss",
        max_iter=1_000,
        use_legacy_attributes=False,
    )
    gnb = GaussianNB()
    svc = NaivelyCalibratedLinearSVC(C=1.0, random_state=RANDOM_STATE)
    rfc = RandomForestClassifier(random_state=RANDOM_STATE)

    return (
        (lr, "Logistic Regression"),
        (gnb, "Naive Bayes"),
        (svc, "SVC"),
        (rfc, "Random forest"),
    )


# ---------------------------------------------------------------------------
# Plotting helper for xorq deferred plot
# ---------------------------------------------------------------------------


@curry
def _calibration_plot_with_histograms(df, title):
    """Build calibration plot with histograms for all classifiers.

    df must have columns: clf_name, y_true, y_prob

    Returns matplotlib Figure matching sklearn example layout.
    """
    # Group by classifier
    clf_names = df[CLF_NAME_COL].unique()
    colors_map = plt.get_cmap("Dark2")
    markers = ("^", "v", "s", "o")

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)

    # Main calibration curve plot
    ax_calibration_curve = fig.add_subplot(gs[:2, :2])

    for i, clf_name in enumerate(clf_names):
        clf_df = df[df[CLF_NAME_COL] == clf_name]
        y_true = clf_df[Y_TRUE_COL].values
        y_prob = clf_df[Y_PROB_COL].values

        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=N_BINS, strategy="uniform"
        )

        ax_calibration_curve.plot(
            prob_pred,
            prob_true,
            marker=markers[i],
            label=clf_name,
            color=colors_map(i),
            linewidth=2,
        )

    ax_calibration_curve.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax_calibration_curve.grid()
    ax_calibration_curve.set_title(title)
    ax_calibration_curve.set_xlabel("Mean predicted probability")
    ax_calibration_curve.set_ylabel("Fraction of positives")
    ax_calibration_curve.legend(loc="lower right")

    # Add histograms
    grid_positions = ((2, 0), (2, 1), (3, 0), (3, 1))
    for i, clf_name in enumerate(clf_names):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        clf_df = df[df[CLF_NAME_COL] == clf_name]
        y_prob = clf_df[Y_PROB_COL].values

        ax.hist(
            y_prob,
            range=(0, 1),
            bins=10,
            label=clf_name,
            color=colors_map(i),
        )
        ax.set(title=clf_name, xlabel="Mean predicted probability", ylabel="Count")

    fig.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit/predict_proba, CalibrationDisplay
# =========================================================================


def sklearn_way(train_df, test_df, feature_cols, clf_list):
    """Eager sklearn: fit classifiers with small training set, compute
    calibration curves on large test set.

    Returns dict with calibration_displays and test data for plotting.
    """
    X_train = train_df[list(feature_cols)].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[list(feature_cols)].values
    y_test = test_df[TARGET_COL].values

    colors = plt.get_cmap("Dark2")
    markers = ("^", "v", "s", "o")

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)
    ax_calibration_curve = fig.add_subplot(gs[:2, :2])

    calibration_displays = {
        name: CalibrationDisplay.from_estimator(
            clf.fit(X_train, y_train),
            X_test,
            y_test,
            n_bins=N_BINS,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
            marker=markers[i],
        )
        for i, (clf, name) in enumerate(clf_list)
    }

    brier_scores = {}
    for name, display in calibration_displays.items():
        brier = brier_score_loss(y_test, display.y_prob)
        brier_scores[name] = brier
        print(f"  sklearn: {name:25s} | Brier = {brier:.4f}")

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")

    # Add histograms
    grid_positions = ((2, 0), (2, 1), (3, 0), (3, 1))
    for i, (name, display) in enumerate(calibration_displays.items()):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])
        ax.hist(
            display.y_prob,
            range=(0, 1),
            bins=10,
            label=name,
            color=colors(i),
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    fig.tight_layout()

    return {
        "fig": fig,
        "calibration_displays": calibration_displays,
        "brier_scores": brier_scores,
    }


# =========================================================================
# XORQ WAY -- deferred calibration curve computation
# =========================================================================


def xorq_way(train_data, test_data, feature_cols, clf_list):
    """Deferred xorq: wrap classifiers in Pipeline.from_instance, deferred
    fit/predict_proba, deferred Brier score.

    Returns dict with classifier name keys and (proba_expr, metrics_expr) values.
    """
    PROBA_COL = "predict_proba"
    make_metric = deferred_sklearn_metric(target=TARGET_COL, pred=PROBA_COL)

    results = {}
    for clf, name in clf_list:
        safe_name = name.replace(" ", "_").lower()
        sklearn_pipe = SklearnPipeline([(safe_name, clf)])
        xorq_pipe = Pipeline.from_instance(sklearn_pipe)
        fitted = xorq_pipe.fit(train_data, features=feature_cols, target=TARGET_COL)
        proba_expr = fitted.predict_proba(test_data)
        metrics = proba_expr.agg(brier=make_metric(metric=brier_score_loss))
        results[name] = (proba_expr, metrics)

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    print("Loading data...")
    df, feature_cols = _load_data()
    clf_list = _build_classifiers()
    classifier_names = tuple(name for _, name in clf_list)

    # Hash-based split via xorq — single source of truth for both paths
    con = xo.connect()
    table = con.register(df, "calibration_compare_data")
    test_size = (N_SAMPLES - TRAIN_SAMPLES) / N_SAMPLES
    train_data, test_data = xo.train_test_splits(
        table,
        test_sizes=test_size,
        unique_key=ROW_IDX,
        random_seed=RANDOM_STATE,
    )

    # Sort by ROW_IDX for deterministic ordering in both paths
    train_data = train_data.order_by(ROW_IDX)
    test_data = test_data.order_by(ROW_IDX)

    # Materialize for sklearn
    train_df = train_data.execute()
    test_df = test_data.execute()

    print("\n=== SKLEARN WAY ===")
    sk_results = sklearn_way(train_df, test_df, feature_cols, clf_list)

    print("\n=== XORQ WAY ===")
    xorq_clf_list = _build_classifiers()
    xo_results = xorq_way(train_data, test_data, feature_cols, xorq_clf_list)

    for name in classifier_names:
        print(f"  xorq:    {name:25s} | deferred")

    print("\n=== ASSERTIONS ===")
    xo_metrics_executed = {
        name: metrics_expr.execute() for name, (_, metrics_expr) in xo_results.items()
    }

    for name in classifier_names:
        xo_brier = xo_metrics_executed[name]["brier"].iloc[0]
        print(f"  xorq:    {name:25s} | Brier = {xo_brier:.4f}")

    sk_brier_df = pd.DataFrame(
        {name: [brier] for name, brier in sk_results["brier_scores"].items()}
    )
    xo_brier_df = pd.DataFrame(
        {
            name: [xo_metrics_executed[name]["brier"].iloc[0]]
            for name in classifier_names
        }
    )
    pd.testing.assert_frame_equal(sk_brier_df, xo_brier_df, rtol=1e-1)
    print("Assertions passed: sklearn and xorq Brier scores match.")

    print("\n=== PLOTTING ===")
    xo_predictions_executed = {
        name: proba_expr.execute().sort_values(ROW_IDX).reset_index(drop=True)
        for name, (proba_expr, _) in xo_results.items()
    }

    xo_plot_dfs = []
    for name in classifier_names:
        pred_df = xo_predictions_executed[name]
        xo_plot_dfs.append(
            pd.DataFrame(
                {
                    Y_TRUE_COL: pred_df[TARGET_COL].values,
                    Y_PROB_COL: np.vstack(pred_df["predict_proba"].values)[:, 1],
                    CLF_NAME_COL: name,
                }
            )
        )
    xo_combined_df = pd.concat(xo_plot_dfs, ignore_index=True)

    xo_plot_table = con.register(xo_combined_df, "calibration_plot_data")

    xo_png = deferred_matplotlib_plot(
        xo_plot_table,
        _calibration_plot_with_histograms(title="Calibration plots"),
    ).execute()

    sk_fig = sk_results["fig"]
    xo_img = load_plot_bytes(xo_png)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    sk_img = fig_to_image(sk_fig)
    axes[0].imshow(sk_img)
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(xo_img)
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(
        "Comparison of Calibration of Classifiers: sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    out = "imgs/plot_compare_calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Composite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
