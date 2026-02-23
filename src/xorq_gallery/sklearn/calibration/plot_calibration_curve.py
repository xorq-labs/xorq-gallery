"""Probability Calibration Curves
==================================

sklearn: Compare calibration of different classifiers (Logistic Regression, GaussianNB
with isotonic/sigmoid calibration via CalibratedClassifierCV) on binary classification
data with 100,000 samples. Evaluate via calibration curves and Brier score to demonstrate
how calibration improves probability estimates without changing predictions.

xorq: Demonstrates xorq's deferred metric aggregation on calibration data. Model fitting
uses sklearn directly (for predict_proba), then xorq's deferred execution computes Brier
scores efficiently. Shows hybrid sklearn/xorq workflow for calibration analysis.

Both produce identical Brier scores proving calibration effectiveness.

Dataset: make_classification (sklearn synthetic, 100k samples, 20 features)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline as SklearnPipeline
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
N_BINS = 10
TEST_SIZE = 0.99  # Use 99% for test, matching sklearn example


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Generate synthetic binary classification dataset.

    Matches sklearn example: 100k samples, 20 features (2 informative, 10 redundant).
    """
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=2,
        n_redundant=10,
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
    - Logistic Regression (baseline, well-calibrated by default)
    - GaussianNB (uncalibrated)
    - GaussianNB + Isotonic calibration
    - GaussianNB + Sigmoid calibration
    """
    lr = LogisticRegression(C=1.0)
    gnb = GaussianNB()
    gnb_isotonic = CalibratedClassifierCV(gnb, cv=2, method="isotonic")
    gnb_sigmoid = CalibratedClassifierCV(gnb, cv=2, method="sigmoid")

    return [
        (lr, "Logistic"),
        (gnb, "Naive Bayes"),
        (gnb_isotonic, "Naive Bayes + Isotonic"),
        (gnb_sigmoid, "Naive Bayes + Sigmoid"),
    ]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _build_sklearn_plot(results_list):
    """Build calibration plot for sklearn results.

    results_list: list of dicts with keys: name, y_prob, y_test, brier
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.get_cmap("Dark2")

    # Plot result 0
    result_0 = results_list[0]
    from sklearn.calibration import calibration_curve
    prob_true_0, prob_pred_0 = calibration_curve(
        result_0["y_test"], result_0["y_prob"], n_bins=N_BINS, strategy="uniform"
    )
    ax.plot(
        prob_pred_0,
        prob_true_0,
        marker="o",
        label=f'{result_0["name"]} (Brier={result_0["brier"]:.3f})',
        color=colors(0),
        linewidth=2,
    )

    # Plot result 1
    result_1 = results_list[1]
    prob_true_1, prob_pred_1 = calibration_curve(
        result_1["y_test"], result_1["y_prob"], n_bins=N_BINS, strategy="uniform"
    )
    ax.plot(
        prob_pred_1,
        prob_true_1,
        marker="o",
        label=f'{result_1["name"]} (Brier={result_1["brier"]:.3f})',
        color=colors(1),
        linewidth=2,
    )

    # Plot result 2
    result_2 = results_list[2]
    prob_true_2, prob_pred_2 = calibration_curve(
        result_2["y_test"], result_2["y_prob"], n_bins=N_BINS, strategy="uniform"
    )
    ax.plot(
        prob_pred_2,
        prob_true_2,
        marker="o",
        label=f'{result_2["name"]} (Brier={result_2["brier"]:.3f})',
        color=colors(2),
        linewidth=2,
    )

    # Plot result 3
    result_3 = results_list[3]
    prob_true_3, prob_pred_3 = calibration_curve(
        result_3["y_test"], result_3["y_prob"], n_bins=N_BINS, strategy="uniform"
    )
    ax.plot(
        prob_pred_3,
        prob_true_3,
        marker="o",
        label=f'{result_3["name"]} (Brier={result_3["brier"]:.3f})',
        color=colors(3),
        linewidth=2,
    )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration plots (Naive Bayes)")
    ax.legend(loc="lower right")
    ax.grid(True)

    plt.tight_layout()
    return fig


def _build_xorq_plot(xo_results_executed):
    """Build calibration plot for xorq results."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.get_cmap("Dark2")

    # Plot result 0
    result_0 = xo_results_executed[0]
    from sklearn.calibration import calibration_curve
    prob_true_0, prob_pred_0 = calibration_curve(
        result_0["y_test"], result_0["y_prob"], n_bins=N_BINS, strategy="uniform"
    )
    ax.plot(
        prob_pred_0,
        prob_true_0,
        marker="o",
        label=f'{result_0["name"]} (Brier={result_0["brier"]:.3f})',
        color=colors(0),
        linewidth=2,
    )

    # Plot result 1
    result_1 = xo_results_executed[1]
    prob_true_1, prob_pred_1 = calibration_curve(
        result_1["y_test"], result_1["y_prob"], n_bins=N_BINS, strategy="uniform"
    )
    ax.plot(
        prob_pred_1,
        prob_true_1,
        marker="o",
        label=f'{result_1["name"]} (Brier={result_1["brier"]:.3f})',
        color=colors(1),
        linewidth=2,
    )

    # Plot result 2
    result_2 = xo_results_executed[2]
    prob_true_2, prob_pred_2 = calibration_curve(
        result_2["y_test"], result_2["y_prob"], n_bins=N_BINS, strategy="uniform"
    )
    ax.plot(
        prob_pred_2,
        prob_true_2,
        marker="o",
        label=f'{result_2["name"]} (Brier={result_2["brier"]:.3f})',
        color=colors(2),
        linewidth=2,
    )

    # Plot result 3
    result_3 = xo_results_executed[3]
    prob_true_3, prob_pred_3 = calibration_curve(
        result_3["y_test"], result_3["y_prob"], n_bins=N_BINS, strategy="uniform"
    )
    ax.plot(
        prob_pred_3,
        prob_true_3,
        marker="o",
        label=f'{result_3["name"]} (Brier={result_3["brier"]:.3f})',
        color=colors(3),
        linewidth=2,
    )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration plots (Naive Bayes)")
    ax.legend(loc="lower right")
    ax.grid(True)

    plt.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager fit/predict_proba, calibration curve computation
# =========================================================================


def sklearn_way(df, feature_cols, clf_list):
    """Eager sklearn: fit classifiers, compute predicted probabilities,
    calculate Brier scores. Matches sklearn example logic 100%.

    Returns list of dicts with name, y_test, y_prob, brier for each classifier.
    """
    X = df[feature_cols].values
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Fit clf 0
    clf_0, name_0 = clf_list[0]
    clf_0.fit(X_train, y_train)
    y_prob_0 = clf_0.predict_proba(X_test)[:, 1]
    brier_0 = brier_score_loss(y_test, y_prob_0)
    print(f"  sklearn: {name_0:30s} | Brier = {brier_0:.4f}")
    result_0 = {
        "name": name_0,
        "y_test": y_test,
        "y_prob": y_prob_0,
        "brier": brier_0,
    }

    # Fit clf 1
    clf_1, name_1 = clf_list[1]
    clf_1.fit(X_train, y_train)
    y_prob_1 = clf_1.predict_proba(X_test)[:, 1]
    brier_1 = brier_score_loss(y_test, y_prob_1)
    print(f"  sklearn: {name_1:30s} | Brier = {brier_1:.4f}")
    result_1 = {
        "name": name_1,
        "y_test": y_test,
        "y_prob": y_prob_1,
        "brier": brier_1,
    }

    # Fit clf 2
    clf_2, name_2 = clf_list[2]
    clf_2.fit(X_train, y_train)
    y_prob_2 = clf_2.predict_proba(X_test)[:, 1]
    brier_2 = brier_score_loss(y_test, y_prob_2)
    print(f"  sklearn: {name_2:30s} | Brier = {brier_2:.4f}")
    result_2 = {
        "name": name_2,
        "y_test": y_test,
        "y_prob": y_prob_2,
        "brier": brier_2,
    }

    # Fit clf 3
    clf_3, name_3 = clf_list[3]
    clf_3.fit(X_train, y_train)
    y_prob_3 = clf_3.predict_proba(X_test)[:, 1]
    brier_3 = brier_score_loss(y_test, y_prob_3)
    print(f"  sklearn: {name_3:30s} | Brier = {brier_3:.4f}")
    result_3 = {
        "name": name_3,
        "y_test": y_test,
        "y_prob": y_prob_3,
        "brier": brier_3,
    }

    return [result_0, result_1, result_2, result_3]


# =========================================================================
# XORQ WAY -- deferred fit/predict_proba, deferred calibration computation
# =========================================================================


def xorq_way(df, feature_cols, fitted_clf_list):
    """Deferred xorq: use xorq for deferred metric computation on calibration data.

    Returns list of dicts with name, metrics_expr, predictions for each classifier.
    Note: Classifiers must be pre-fitted (done in main()).
    """
    con = xo.connect()

    # Split data matching sklearn's train_test_split
    X = df[feature_cols].values
    y = df["y"].values

    from sklearn.model_selection import train_test_split as sklearn_split

    X_train, X_test, y_train, y_test = sklearn_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Clf 0
    clf_0, name_0 = fitted_clf_list[0]
    y_prob_0 = clf_0.predict_proba(X_test)[:, 1]
    prob_df_0 = pd.DataFrame({
        "y": y_test,
        "prob_pos": y_prob_0,
    })
    prob_table_0 = con.register(prob_df_0, f"probs_{name_0.replace(' ', '_')}")
    make_metric_0 = deferred_sklearn_metric(target="y", pred="prob_pos")
    metrics_0 = prob_table_0.agg(brier=make_metric_0(metric=brier_score_loss))
    result_0 = {
        "name": name_0,
        "metrics": metrics_0,
        "predictions": prob_table_0,
    }

    # Clf 1
    clf_1, name_1 = fitted_clf_list[1]
    y_prob_1 = clf_1.predict_proba(X_test)[:, 1]
    prob_df_1 = pd.DataFrame({
        "y": y_test,
        "prob_pos": y_prob_1,
    })
    prob_table_1 = con.register(prob_df_1, f"probs_{name_1.replace(' ', '_')}")
    make_metric_1 = deferred_sklearn_metric(target="y", pred="prob_pos")
    metrics_1 = prob_table_1.agg(brier=make_metric_1(metric=brier_score_loss))
    result_1 = {
        "name": name_1,
        "metrics": metrics_1,
        "predictions": prob_table_1,
    }

    # Clf 2
    clf_2, name_2 = fitted_clf_list[2]
    y_prob_2 = clf_2.predict_proba(X_test)[:, 1]
    prob_df_2 = pd.DataFrame({
        "y": y_test,
        "prob_pos": y_prob_2,
    })
    prob_table_2 = con.register(prob_df_2, f"probs_{name_2.replace(' ', '_')}")
    make_metric_2 = deferred_sklearn_metric(target="y", pred="prob_pos")
    metrics_2 = prob_table_2.agg(brier=make_metric_2(metric=brier_score_loss))
    result_2 = {
        "name": name_2,
        "metrics": metrics_2,
        "predictions": prob_table_2,
    }

    # Clf 3
    clf_3, name_3 = fitted_clf_list[3]
    y_prob_3 = clf_3.predict_proba(X_test)[:, 1]
    prob_df_3 = pd.DataFrame({
        "y": y_test,
        "prob_pos": y_prob_3,
    })
    prob_table_3 = con.register(prob_df_3, f"probs_{name_3.replace(' ', '_')}")
    make_metric_3 = deferred_sklearn_metric(target="y", pred="prob_pos")
    metrics_3 = prob_table_3.agg(brier=make_metric_3(metric=brier_score_loss))
    result_3 = {
        "name": name_3,
        "metrics": metrics_3,
        "predictions": prob_table_3,
    }

    return [result_0, result_1, result_2, result_3]


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
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    xorq_clf_list = _build_classifiers()
    fitted_xorq_clfs = []

    clf_0, name_0 = xorq_clf_list[0]
    sklearn_pipe_0 = SklearnPipeline([("clf", clf_0)])
    sklearn_pipe_0.fit(X_train, y_train)
    fitted_xorq_clfs.append((sklearn_pipe_0, name_0))

    clf_1, name_1 = xorq_clf_list[1]
    sklearn_pipe_1 = SklearnPipeline([("clf", clf_1)])
    sklearn_pipe_1.fit(X_train, y_train)
    fitted_xorq_clfs.append((sklearn_pipe_1, name_1))

    clf_2, name_2 = xorq_clf_list[2]
    sklearn_pipe_2 = SklearnPipeline([("clf", clf_2)])
    sklearn_pipe_2.fit(X_train, y_train)
    fitted_xorq_clfs.append((sklearn_pipe_2, name_2))

    clf_3, name_3 = xorq_clf_list[3]
    sklearn_pipe_3 = SklearnPipeline([("clf", clf_3)])
    sklearn_pipe_3.fit(X_train, y_train)
    fitted_xorq_clfs.append((sklearn_pipe_3, name_3))

    xo_results = xorq_way(df, feature_cols, fitted_xorq_clfs)

    # Execute deferred metrics and assert equivalence
    print("\n=== ASSERTIONS ===")

    # Result 0
    sk_result_0 = sk_results[0]
    xo_result_0 = xo_results[0]
    sk_brier_0 = sk_result_0["brier"]
    xo_metrics_df_0 = xo_result_0["metrics"].execute()
    xo_brier_0 = xo_metrics_df_0["brier"].iloc[0]
    print(f"  xorq:   {xo_result_0['name']:30s} | Brier = {xo_brier_0:.4f}")
    np.testing.assert_allclose(sk_brier_0, xo_brier_0, rtol=1e-2)

    # Result 1
    sk_result_1 = sk_results[1]
    xo_result_1 = xo_results[1]
    sk_brier_1 = sk_result_1["brier"]
    xo_metrics_df_1 = xo_result_1["metrics"].execute()
    xo_brier_1 = xo_metrics_df_1["brier"].iloc[0]
    print(f"  xorq:   {xo_result_1['name']:30s} | Brier = {xo_brier_1:.4f}")
    np.testing.assert_allclose(sk_brier_1, xo_brier_1, rtol=1e-2)

    # Result 2
    sk_result_2 = sk_results[2]
    xo_result_2 = xo_results[2]
    sk_brier_2 = sk_result_2["brier"]
    xo_metrics_df_2 = xo_result_2["metrics"].execute()
    xo_brier_2 = xo_metrics_df_2["brier"].iloc[0]
    print(f"  xorq:   {xo_result_2['name']:30s} | Brier = {xo_brier_2:.4f}")
    np.testing.assert_allclose(sk_brier_2, xo_brier_2, rtol=1e-2)

    # Result 3
    sk_result_3 = sk_results[3]
    xo_result_3 = xo_results[3]
    sk_brier_3 = sk_result_3["brier"]
    xo_metrics_df_3 = xo_result_3["metrics"].execute()
    xo_brier_3 = xo_metrics_df_3["brier"].iloc[0]
    print(f"  xorq:   {xo_result_3['name']:30s} | Brier = {xo_brier_3:.4f}")
    np.testing.assert_allclose(sk_brier_3, xo_brier_3, rtol=1e-2)

    print("Assertions passed: sklearn and xorq Brier scores match.")

    # Build sklearn calibration plot
    print("\n=== PLOTTING ===")
    sk_fig = _build_sklearn_plot(sk_results)

    # Execute xorq predictions
    preds_df_0 = xo_result_0["predictions"].execute()
    xo_executed_0 = {
        "name": xo_result_0["name"],
        "y_test": preds_df_0["y"].values,
        "y_prob": preds_df_0["prob_pos"].values,
        "brier": xo_result_0["metrics"].execute()["brier"].iloc[0],
    }

    preds_df_1 = xo_result_1["predictions"].execute()
    xo_executed_1 = {
        "name": xo_result_1["name"],
        "y_test": preds_df_1["y"].values,
        "y_prob": preds_df_1["prob_pos"].values,
        "brier": xo_result_1["metrics"].execute()["brier"].iloc[0],
    }

    preds_df_2 = xo_result_2["predictions"].execute()
    xo_executed_2 = {
        "name": xo_result_2["name"],
        "y_test": preds_df_2["y"].values,
        "y_prob": preds_df_2["prob_pos"].values,
        "brier": xo_result_2["metrics"].execute()["brier"].iloc[0],
    }

    preds_df_3 = xo_result_3["predictions"].execute()
    xo_executed_3 = {
        "name": xo_result_3["name"],
        "y_test": preds_df_3["y"].values,
        "y_prob": preds_df_3["prob_pos"].values,
        "brier": xo_result_3["metrics"].execute()["brier"].iloc[0],
    }

    xo_results_executed = [xo_executed_0, xo_executed_1, xo_executed_2, xo_executed_3]

    xo_fig = _build_xorq_plot(xo_results_executed)

    # Create composite plot: sklearn (left) | xorq (right)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: sklearn plot
    sk_img = fig_to_image(sk_fig)
    axes[0].imshow(sk_img)
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Right: xorq plot
    xo_img = fig_to_image(xo_fig)
    axes[1].imshow(xo_img)
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    plt.close(xo_fig)
    plt.close(sk_fig)

    # Save composite
    fig.suptitle(
        "Probability Calibration Curves: sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    out = "imgs/plot_calibration_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Composite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
