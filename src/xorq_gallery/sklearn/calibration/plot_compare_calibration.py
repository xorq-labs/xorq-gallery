"""Comparison of Calibration of Classifiers
========================================

sklearn: Compare calibration of four different classifiers (Logistic Regression,
Naive Bayes, Linear SVC, Random Forest) on binary classification with 100,000
samples (100 train, 99,900 test). Uses CalibrationDisplay to plot calibration
curves and histograms showing probability distributions.

xorq: Demonstrates xorq's deferred execution for calibration analysis. Models
are fit eagerly with sklearn (for predict_proba), then xorq computes deferred
calibration curves and histograms. Shows hybrid sklearn/xorq workflow for
multi-model calibration comparison.

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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.svm import LinearSVC
from xorq.expr.ml.metrics import deferred_sklearn_metric

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
    feature_cols = [f"x{i}" for i in range(N_FEATURES)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["y"] = y
    df["row_idx"] = range(len(df))

    return df, feature_cols


def _build_classifiers():
    """Return list of (clf, name) tuples matching sklearn example.

    Includes:
    - Logistic Regression with CV (auto-tuned regularization)
    - Gaussian Naive Bayes
    - Linear SVC with naive probability calibration
    - Random Forest
    """
    lr = LogisticRegressionCV(
        Cs=np.logspace(-6, 6, 101),
        cv=10,
        l1_ratios=(0,),
        scoring="neg_log_loss",
        max_iter=1_000,
        use_legacy_attributes=False,
    )
    gnb = GaussianNB()
    svc = NaivelyCalibratedLinearSVC(C=1.0, random_state=RANDOM_STATE)
    rfc = RandomForestClassifier(random_state=RANDOM_STATE)

    return [
        (lr, "Logistic Regression"),
        (gnb, "Naive Bayes"),
        (svc, "SVC"),
        (rfc, "Random forest"),
    ]


# ---------------------------------------------------------------------------
# Plotting helper for xorq deferred plot
# ---------------------------------------------------------------------------


def _build_calibration_plot_with_histograms(df):
    """Build calibration plot with histograms for all classifiers.

    df must have columns: clf_name, y_true, y_prob

    Returns matplotlib Figure matching sklearn example layout.
    """
    # Group by classifier
    clf_names = df["clf_name"].unique()
    colors_map = plt.get_cmap("Dark2")
    markers = ["^", "v", "s", "o"]

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)

    # Main calibration curve plot
    ax_calibration_curve = fig.add_subplot(gs[:2, :2])

    for i, clf_name in enumerate(clf_names):
        clf_df = df[df["clf_name"] == clf_name]
        y_true = clf_df["y_true"].values
        y_prob = clf_df["y_prob"].values

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
    ax_calibration_curve.set_title("Calibration plots")
    ax_calibration_curve.set_xlabel("Mean predicted probability")
    ax_calibration_curve.set_ylabel("Fraction of positives")
    ax_calibration_curve.legend(loc="lower right")

    # Add histograms
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
    for i, clf_name in enumerate(clf_names):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        clf_df = df[df["clf_name"] == clf_name]
        y_prob = clf_df["y_prob"].values

        ax.hist(
            y_prob,
            range=(0, 1),
            bins=10,
            label=clf_name,
            color=colors_map(i),
        )
        ax.set(title=clf_name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit/predict_proba, CalibrationDisplay
# =========================================================================


def sklearn_way(df, feature_cols, clf_list):
    """Eager sklearn: fit classifiers with small training set, compute
    calibration curves on large test set. Matches sklearn example logic 100%.

    Returns dict with calibration_displays and test data for plotting.
    """
    X = df[feature_cols].values
    y = df["y"].values

    # Split: 100 train, 99,900 test (matching sklearn example)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        shuffle=False,
        test_size=N_SAMPLES - TRAIN_SAMPLES,
    )

    # Fit each classifier and create CalibrationDisplay
    calibration_displays = {}
    colors = plt.get_cmap("Dark2")
    markers = ["^", "v", "s", "o"]

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)
    ax_calibration_curve = fig.add_subplot(gs[:2, :2])

    # Fit clf 0
    clf_0, name_0 = clf_list[0]
    clf_0.fit(X_train, y_train)
    display_0 = CalibrationDisplay.from_estimator(
        clf_0,
        X_test,
        y_test,
        n_bins=N_BINS,
        name=name_0,
        ax=ax_calibration_curve,
        color=colors(0),
        marker=markers[0],
    )
    calibration_displays[name_0] = display_0
    print(f"  sklearn: {name_0:25s} | fitted")

    # Fit clf 1
    clf_1, name_1 = clf_list[1]
    clf_1.fit(X_train, y_train)
    display_1 = CalibrationDisplay.from_estimator(
        clf_1,
        X_test,
        y_test,
        n_bins=N_BINS,
        name=name_1,
        ax=ax_calibration_curve,
        color=colors(1),
        marker=markers[1],
    )
    calibration_displays[name_1] = display_1
    print(f"  sklearn: {name_1:25s} | fitted")

    # Fit clf 2
    clf_2, name_2 = clf_list[2]
    clf_2.fit(X_train, y_train)
    display_2 = CalibrationDisplay.from_estimator(
        clf_2,
        X_test,
        y_test,
        n_bins=N_BINS,
        name=name_2,
        ax=ax_calibration_curve,
        color=colors(2),
        marker=markers[2],
    )
    calibration_displays[name_2] = display_2
    print(f"  sklearn: {name_2:25s} | fitted")

    # Fit clf 3
    clf_3, name_3 = clf_list[3]
    clf_3.fit(X_train, y_train)
    display_3 = CalibrationDisplay.from_estimator(
        clf_3,
        X_test,
        y_test,
        n_bins=N_BINS,
        name=name_3,
        ax=ax_calibration_curve,
        color=colors(3),
        marker=markers[3],
    )
    calibration_displays[name_3] = display_3
    print(f"  sklearn: {name_3:25s} | fitted")

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")

    # Add histograms
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]

    # Histogram 0
    row_0, col_0 = grid_positions[0]
    ax_0 = fig.add_subplot(gs[row_0, col_0])
    ax_0.hist(
        calibration_displays[name_0].y_prob,
        range=(0, 1),
        bins=10,
        label=name_0,
        color=colors(0),
    )
    ax_0.set(title=name_0, xlabel="Mean predicted probability", ylabel="Count")

    # Histogram 1
    row_1, col_1 = grid_positions[1]
    ax_1 = fig.add_subplot(gs[row_1, col_1])
    ax_1.hist(
        calibration_displays[name_1].y_prob,
        range=(0, 1),
        bins=10,
        label=name_1,
        color=colors(1),
    )
    ax_1.set(title=name_1, xlabel="Mean predicted probability", ylabel="Count")

    # Histogram 2
    row_2, col_2 = grid_positions[2]
    ax_2 = fig.add_subplot(gs[row_2, col_2])
    ax_2.hist(
        calibration_displays[name_2].y_prob,
        range=(0, 1),
        bins=10,
        label=name_2,
        color=colors(2),
    )
    ax_2.set(title=name_2, xlabel="Mean predicted probability", ylabel="Count")

    # Histogram 3
    row_3, col_3 = grid_positions[3]
    ax_3 = fig.add_subplot(gs[row_3, col_3])
    ax_3.hist(
        calibration_displays[name_3].y_prob,
        range=(0, 1),
        bins=10,
        label=name_3,
        color=colors(3),
    )
    ax_3.set(title=name_3, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()

    return {
        "fig": fig,
        "calibration_displays": calibration_displays,
        "clf_list": clf_list,
    }


# =========================================================================
# XORQ WAY -- deferred calibration curve computation
# =========================================================================


def xorq_way(df, feature_cols, fitted_clf_list):
    """Deferred xorq: use xorq for deferred calibration analysis.

    Returns dict with deferred plot expression.
    Note: Classifiers must be pre-fitted (done in main()).
    """
    con = xo.connect()

    # Split data matching sklearn's train_test_split
    X = df[feature_cols].values
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        shuffle=False,
        test_size=N_SAMPLES - TRAIN_SAMPLES,
    )

    # Collect predictions from pre-fitted classifiers
    clf_0, name_0 = fitted_clf_list[0]
    y_prob_0 = clf_0.predict_proba(X_test)[:, 1]
    clf_preds_0 = pd.DataFrame({
        "y_true": y_test,
        "y_prob": y_prob_0,
        "clf_name": name_0,
    })

    clf_1, name_1 = fitted_clf_list[1]
    y_prob_1 = clf_1.predict_proba(X_test)[:, 1]
    clf_preds_1 = pd.DataFrame({
        "y_true": y_test,
        "y_prob": y_prob_1,
        "clf_name": name_1,
    })

    clf_2, name_2 = fitted_clf_list[2]
    y_prob_2 = clf_2.predict_proba(X_test)[:, 1]
    clf_preds_2 = pd.DataFrame({
        "y_true": y_test,
        "y_prob": y_prob_2,
        "clf_name": name_2,
    })

    clf_3, name_3 = fitted_clf_list[3]
    y_prob_3 = clf_3.predict_proba(X_test)[:, 1]
    clf_preds_3 = pd.DataFrame({
        "y_true": y_test,
        "y_prob": y_prob_3,
        "clf_name": name_3,
    })

    # Combine all predictions into one dataframe
    combined_df = pd.concat([clf_preds_0, clf_preds_1, clf_preds_2, clf_preds_3], ignore_index=True)

    # Register as xorq table for deferred plotting
    predictions_table = con.register(combined_df, "calibration_predictions")

    return {
        "predictions_table": predictions_table,
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    print("Loading data...")
    df, feature_cols = _load_data()
    clf_list = _build_classifiers()

    print("\n=== SKLEARN WAY ===")
    sk_results = sklearn_way(df, feature_cols, clf_list)

    print("\n=== XORQ WAY ===")
    # Build fresh classifiers and fit them for xorq
    X = df[feature_cols].values
    y = df["y"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        shuffle=False,
        test_size=N_SAMPLES - TRAIN_SAMPLES,
    )

    xorq_clf_list = _build_classifiers()
    fitted_xorq_clfs = []
    clf_0, name_0 = xorq_clf_list[0]
    clf_0.fit(X_train, y_train)
    fitted_xorq_clfs.append((clf_0, name_0))
    print(f"  xorq:    {name_0:25s} | fitted")

    clf_1, name_1 = xorq_clf_list[1]
    clf_1.fit(X_train, y_train)
    fitted_xorq_clfs.append((clf_1, name_1))
    print(f"  xorq:    {name_1:25s} | fitted")

    clf_2, name_2 = xorq_clf_list[2]
    clf_2.fit(X_train, y_train)
    fitted_xorq_clfs.append((clf_2, name_2))
    print(f"  xorq:    {name_2:25s} | fitted")

    clf_3, name_3 = xorq_clf_list[3]
    clf_3.fit(X_train, y_train)
    fitted_xorq_clfs.append((clf_3, name_3))
    print(f"  xorq:    {name_3:25s} | fitted")

    xo_results = xorq_way(df, feature_cols, fitted_xorq_clfs)

    # Execute deferred plot in main()
    print("\n=== EXECUTING DEFERRED PLOT ===")
    xo_png = deferred_matplotlib_plot(
        xo_results["predictions_table"], _build_calibration_plot_with_histograms, name="plot"
    ).execute()

    # Extract calibration data from both for assertion
    print("\n=== ASSERTIONS ===")
    # We'll compare the y_prob arrays for each classifier
    sk_displays = sk_results["calibration_displays"]

    # Execute xorq predictions to compare
    xo_preds_df = xo_results["predictions_table"].execute()

    name_0_check = fitted_xorq_clfs[0][1]
    sk_y_prob_0 = sk_displays[name_0_check].y_prob
    xo_clf_df_0 = xo_preds_df[xo_preds_df["clf_name"] == name_0_check]
    xo_y_prob_0 = xo_clf_df_0["y_prob"].values
    np.testing.assert_allclose(sk_y_prob_0, xo_y_prob_0, rtol=1e-10)
    print(f"  {name_0_check:25s} | predictions match")

    name_1_check = fitted_xorq_clfs[1][1]
    sk_y_prob_1 = sk_displays[name_1_check].y_prob
    xo_clf_df_1 = xo_preds_df[xo_preds_df["clf_name"] == name_1_check]
    xo_y_prob_1 = xo_clf_df_1["y_prob"].values
    np.testing.assert_allclose(sk_y_prob_1, xo_y_prob_1, rtol=1e-10)
    print(f"  {name_1_check:25s} | predictions match")

    name_2_check = fitted_xorq_clfs[2][1]
    sk_y_prob_2 = sk_displays[name_2_check].y_prob
    xo_clf_df_2 = xo_preds_df[xo_preds_df["clf_name"] == name_2_check]
    xo_y_prob_2 = xo_clf_df_2["y_prob"].values
    np.testing.assert_allclose(sk_y_prob_2, xo_y_prob_2, rtol=1e-10)
    print(f"  {name_2_check:25s} | predictions match")

    name_3_check = fitted_xorq_clfs[3][1]
    sk_y_prob_3 = sk_displays[name_3_check].y_prob
    xo_clf_df_3 = xo_preds_df[xo_preds_df["clf_name"] == name_3_check]
    xo_y_prob_3 = xo_clf_df_3["y_prob"].values
    np.testing.assert_allclose(sk_y_prob_3, xo_y_prob_3, rtol=1e-10)
    print(f"  {name_3_check:25s} | predictions match")

    print("Assertions passed: sklearn and xorq predictions match.")

    # Build composite plot: sklearn (left) | xorq (right)
    print("\n=== BUILDING COMPOSITE PLOT ===")
    sk_fig = sk_results["fig"]
    xo_img = load_plot_bytes(xo_png)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Left: sklearn plot
    sk_img = fig_to_image(sk_fig)
    axes[0].imshow(sk_img)
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Right: xorq plot
    axes[1].imshow(xo_img)
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Save composite
    fig.suptitle(
        "Comparison of Calibration of Classifiers: sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    out = "imgs/plot_compare_calibration.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Composite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
