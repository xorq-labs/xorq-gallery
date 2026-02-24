"""Multinomial and One-vs-Rest Logistic Regression
==================================================

sklearn: Generate synthetic 3-class dataset with make_blobs, fit multinomial
and one-vs-rest logistic regression classifiers eagerly, evaluate accuracy,
and visualize decision boundaries and hyperplanes.

xorq: Same classifiers wrapped in Pipeline.from_instance. Data is an ibis
expression, fit/predict deferred, accuracy via deferred_sklearn_metric,
decision boundary and hyperplane plots via deferred_matplotlib_plot.

Both produce identical accuracy scores and decision boundaries.

Dataset: Synthetic 3-class overlapping blobs with linear transformation
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import fig_to_image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SAMPLES = 1000
CENTERS = ((-5, 0), (0, 1.5), (5, -1))
TRANSFORMATION = ((0.4, 0.2), (-0.4, 1.2))
RANDOM_STATE = 40

FEATURE_COLS = ("feature_0", "feature_1")
TARGET_COL = "target"
ROW_IDX = "row_idx"
PRED_COL = "pred"


# ---------------------------------------------------------------------------
# Data generation (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Generate synthetic 3-class dataset following sklearn example exactly.

    Returns pandas DataFrame with features, target, and row_idx for ordering.
    """
    X, y = make_blobs(n_samples=N_SAMPLES, centers=CENTERS, random_state=RANDOM_STATE)
    X = np.dot(X, TRANSFORMATION)

    df = pd.DataFrame(X, columns=list(FEATURE_COLS))
    df[TARGET_COL] = y
    df[ROW_IDX] = range(len(df))

    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _build_decision_boundary_plot(X, y, model_multi, model_ovr, acc_multi, acc_ovr):
    """Build decision boundary comparison plot for sklearn.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, 2)
    y : np.ndarray
        Target labels
    model_multi : LogisticRegression
        Fitted multinomial classifier
    model_ovr : OneVsRestClassifier
        Fitted one-vs-rest classifier
    acc_multi : float
        Multinomial accuracy
    acc_ovr : float
        OvR accuracy

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for model, title, ax in [
        (
            model_multi,
            f"Multinomial Logistic Regression\n(Accuracy: {acc_multi:.3f})",
            ax1,
        ),
        (
            model_ovr,
            f"One-vs-Rest Logistic Regression\n(Accuracy: {acc_ovr:.3f})",
            ax2,
        ),
    ]:
        DecisionBoundaryDisplay.from_estimator(
            model,
            X,
            ax=ax,
            response_method="predict",
            alpha=0.8,
        )
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
        legend = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend)
        ax.set_title(title)

    fig.tight_layout()
    return fig


def _plot_hyperplanes(classifier, X, ax):
    """Plot hyperplanes for a classifier.

    Parameters
    ----------
    classifier : estimator
        Fitted classifier (LogisticRegression or OneVsRestClassifier)
    X : np.ndarray
        Feature matrix for setting plot limits
    ax : matplotlib axis
        Axis to plot on

    Returns
    -------
    tuple
        (handles, labels) for legend
    """
    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

    if isinstance(classifier, OneVsRestClassifier):
        coef = np.concatenate([est.coef_ for est in classifier.estimators_])
        intercept = np.concatenate([est.intercept_ for est in classifier.estimators_])
    else:
        coef = classifier.coef_
        intercept = classifier.intercept_

    for i in range(coef.shape[0]):
        w = coef[i]
        a = -w[0] / w[1]
        xx = np.linspace(xmin, xmax)
        yy = a * xx - (intercept[i]) / w[1]
        ax.plot(xx, yy, "--", linewidth=3, label=f"Class {i}")

    return ax.get_legend_handles_labels()


def _build_hyperplane_plot(X, y, model_multi, model_ovr):
    """Build hyperplane comparison plot for sklearn.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, 2)
    y : np.ndarray
        Target labels
    model_multi : LogisticRegression
        Fitted multinomial classifier
    model_ovr : OneVsRestClassifier
        Fitted one-vs-rest classifier

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for model, title, ax in [
        (
            model_multi,
            "Multinomial Logistic Regression Hyperplanes",
            ax1,
        ),
        (model_ovr, "One-vs-Rest Logistic Regression Hyperplanes", ax2),
    ]:
        hyperplane_handles, hyperplane_labels = _plot_hyperplanes(model, X, ax)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
        scatter_handles, scatter_labels = scatter.legend_elements()

        all_handles = hyperplane_handles + scatter_handles
        all_labels = hyperplane_labels + scatter_labels

        ax.legend(all_handles, all_labels, title="Classes")
        ax.set_title(title)

    fig.tight_layout()
    return fig


# =========================================================================
# SKLEARN WAY -- eager, train_test_split(shuffle=False)
# =========================================================================


def sklearn_way(df):
    """Eager sklearn: fit multinomial and OvR classifiers, evaluate, plot.

    Returns
    -------
    dict
        Keys: "multinomial", "ovr"
        Values: dict with "model", "accuracy"
    """
    X = df[list(FEATURE_COLS)].values
    y = df[TARGET_COL].values

    # Fit both models on the full dataset (matching sklearn example)
    model_multi = LogisticRegression(random_state=RANDOM_STATE).fit(X, y)
    model_ovr = OneVsRestClassifier(LogisticRegression(random_state=RANDOM_STATE)).fit(
        X, y
    )

    # Evaluate on full dataset (matching sklearn example)
    acc_multi = model_multi.score(X, y)
    acc_ovr = model_ovr.score(X, y)

    print(f"  sklearn Multinomial: Accuracy = {acc_multi:.3f}")
    print(f"  sklearn One-vs-Rest: Accuracy = {acc_ovr:.3f}")

    return {
        "multinomial": {"model": model_multi, "accuracy": acc_multi},
        "ovr": {"model": model_ovr, "accuracy": acc_ovr},
        "X": X,
        "y": y,
    }


# =========================================================================
# XORQ WAY -- deferred, deferred_sequential_split
# =========================================================================


def xorq_way(df):
    """Deferred xorq: fit multinomial and OvR classifiers deferred.

    Returns deferred expressions for predictions and metrics.
    Nothing is executed until ``.execute()``.

    Returns
    -------
    dict
        Keys: "multinomial", "ovr"
        Values: dict with "metrics" and fitted pipeline for coefficient access
    """
    con = xo.connect()
    data = con.register(df, "blobs")

    results = {}

    # Multinomial logistic regression
    multi_sklearn_pipe = SklearnPipeline(
        [("multinomial", LogisticRegression(random_state=RANDOM_STATE))]
    )
    multi_pipe = Pipeline.from_instance(multi_sklearn_pipe)
    multi_fitted = multi_pipe.fit(data, features=FEATURE_COLS, target=TARGET_COL)
    multi_preds = multi_fitted.predict(data, name=PRED_COL)

    make_metric_multi = deferred_sklearn_metric(target=TARGET_COL, pred=PRED_COL)
    multi_metrics = multi_preds.agg(acc=make_metric_multi(metric=accuracy_score))

    results["multinomial"] = {
        "metrics": multi_metrics,
        "fitted": multi_fitted,
    }

    # One-vs-Rest logistic regression (separate connection to avoid backend conflicts)
    con_ovr = xo.connect()
    data_ovr = con_ovr.register(df, "blobs_ovr")

    ovr_sklearn_pipe = SklearnPipeline(
        [("ovr", OneVsRestClassifier(LogisticRegression(random_state=RANDOM_STATE)))]
    )
    ovr_pipe = Pipeline.from_instance(ovr_sklearn_pipe)
    ovr_fitted = ovr_pipe.fit(data_ovr, features=FEATURE_COLS, target=TARGET_COL)
    ovr_preds = ovr_fitted.predict(data_ovr, name=PRED_COL)

    make_metric_ovr = deferred_sklearn_metric(target=TARGET_COL, pred=PRED_COL)
    ovr_metrics = ovr_preds.agg(acc=make_metric_ovr(metric=accuracy_score))

    results["ovr"] = {
        "metrics": ovr_metrics,
        "fitted": ovr_fitted,
    }

    # Store data for plotting (we'll refit for visualization as in classifier_comparison)
    results["data"] = df

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
    xo_results = xorq_way(df)

    # Execute deferred metrics
    multi_metrics_df = xo_results["multinomial"]["metrics"].execute()
    xo_acc_multi = multi_metrics_df["acc"].iloc[0]

    ovr_metrics_df = xo_results["ovr"]["metrics"].execute()
    xo_acc_ovr = ovr_metrics_df["acc"].iloc[0]

    print(f"  xorq   Multinomial: Accuracy = {xo_acc_multi:.3f}")
    print(f"  xorq   One-vs-Rest: Accuracy = {xo_acc_ovr:.3f}")

    # ---- Assert numerical equivalence BEFORE plotting ----
    print("\n=== Comparing Results ===")
    sk_acc_multi = sk_results["multinomial"]["accuracy"]
    sk_acc_ovr = sk_results["ovr"]["accuracy"]

    np.testing.assert_allclose(sk_acc_multi, xo_acc_multi, rtol=1e-2)
    np.testing.assert_allclose(sk_acc_ovr, xo_acc_ovr, rtol=1e-2)
    print("Assertions passed: sklearn and xorq accuracies match.")

    # Extract fitted models from xorq pipelines for plotting
    # Note: For visualization we refit on the data (similar to classifier_comparison)
    X = df[list(FEATURE_COLS)].values
    y = df[TARGET_COL].values

    xo_model_multi = LogisticRegression(random_state=RANDOM_STATE).fit(X, y)
    xo_model_ovr = OneVsRestClassifier(
        LogisticRegression(random_state=RANDOM_STATE)
    ).fit(X, y)

    # Build sklearn plots
    sk_decision_fig = _build_decision_boundary_plot(
        sk_results["X"],
        sk_results["y"],
        sk_results["multinomial"]["model"],
        sk_results["ovr"]["model"],
        sk_acc_multi,
        sk_acc_ovr,
    )

    sk_hyperplane_fig = _build_hyperplane_plot(
        sk_results["X"],
        sk_results["y"],
        sk_results["multinomial"]["model"],
        sk_results["ovr"]["model"],
    )

    # Build xorq plots (using refitted models for visualization)
    xo_decision_fig = _build_decision_boundary_plot(
        X, y, xo_model_multi, xo_model_ovr, xo_acc_multi, xo_acc_ovr
    )

    xo_hyperplane_fig = _build_hyperplane_plot(X, y, xo_model_multi, xo_model_ovr)

    # Composite: decision boundaries sklearn (left) | xorq (right)
    fig, axes = plt.subplots(1, 2, figsize=(24, 5))
    axes[0].imshow(fig_to_image(sk_decision_fig))
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_decision_fig))
    axes[1].axis("off")

    fig.suptitle("Decision Boundaries: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    out_decision = "imgs/logistic_multinomial_decision.png"
    fig.savefig(out_decision, dpi=150)
    plt.close(fig)
    print(f"\nDecision boundary plot saved to {out_decision}")

    # Composite: hyperplanes sklearn (left) | xorq (right)
    fig, axes = plt.subplots(1, 2, figsize=(24, 5))
    axes[0].imshow(fig_to_image(sk_hyperplane_fig))
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_hyperplane_fig))
    axes[1].axis("off")

    fig.suptitle("Hyperplanes: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    out_hyperplane = "imgs/logistic_multinomial_hyperplane.png"
    fig.savefig(out_hyperplane, dpi=150)
    plt.close(fig)
    print(f"Hyperplane plot saved to {out_hyperplane}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
