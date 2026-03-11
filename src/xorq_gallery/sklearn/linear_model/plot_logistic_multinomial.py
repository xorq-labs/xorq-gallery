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

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/linear_model/plot_logistic_multinomial.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
)
from xorq_gallery.utils import fig_to_image, save_fig


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


def load_data():
    """Generate synthetic 3-class dataset following sklearn example exactly.

    Returns pandas DataFrame with features, target, and row_idx for ordering.
    """
    X, y = make_blobs(n_samples=N_SAMPLES, centers=CENTERS, random_state=RANDOM_STATE)
    X = np.dot(X, TRANSFORMATION)

    df = pd.DataFrame(X, columns=list(FEATURE_COLS)).assign(
        **{TARGET_COL: y, ROW_IDX: range(len(X))}
    )

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


def save_decision_plot(X, y, sk_models, sk_accuracies, xo_models, xo_accuracies):
    sk_fig = _build_decision_boundary_plot(
        X,
        y,
        sk_models[MULTINOMIAL],
        sk_models[OVR],
        sk_accuracies[MULTINOMIAL],
        sk_accuracies[OVR],
    )
    xo_fig = _build_decision_boundary_plot(
        X,
        y,
        xo_models[MULTINOMIAL],
        xo_models[OVR],
        xo_accuracies[MULTINOMIAL],
        xo_accuracies[OVR],
    )

    fig, axes = plt.subplots(1, 2, figsize=(24, 5))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].axis("off")

    fig.suptitle("Decision Boundaries: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    return fig


def save_hyperplane_plot(X, y, sk_models, xo_models):
    sk_fig = _build_hyperplane_plot(X, y, sk_models[MULTINOMIAL], sk_models[OVR])
    xo_fig = _build_hyperplane_plot(X, y, xo_models[MULTINOMIAL], xo_models[OVR])

    fig, axes = plt.subplots(1, 2, figsize=(24, 5))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].axis("off")

    fig.suptitle("Hyperplanes: sklearn vs xorq", fontsize=16)
    fig.tight_layout()
    return fig


def plot_results(comparator):
    sk_accuracies, xo_accuracies = (
        {name: result["metrics"]["acc"] for name, result in results.items()}
        for results in (comparator.sklearn_results, comparator.xorq_results)
    )
    sk_models, xo_models = (
        {name: result["fitted"] for name, result in results.items()}
        for results in (comparator.sklearn_results, comparator.xorq_results)
    )
    X = comparator.df[list(comparator.features)].values
    y = comparator.df[comparator.target].values
    fh0 = save_decision_plot(X, y, sk_models, sk_accuracies, xo_models, xo_accuracies)
    fh1 = save_hyperplane_plot(X, y, sk_models, xo_models)
    return fh0, fh1


def save_comparison_plots(comparator):
    fh0, fh1 = plot_results(comparator)
    save_fig("imgs/logistic_multinomial_decision.png", fh0)
    save_fig("imgs/logistic_multinomial_hyperplane.png", fh1)


def compute_with_xorq(name_to_xorq_exprs):
    """Execute deferred xorq expressions for multinomial and OvR classifiers.

    Returns
    -------
    tuple
        (xo_accuracies, xo_models) dicts keyed by method name
    """
    xo_accuracies = {
        name: exprs["metrics"].execute()["acc"].iloc[0]
        for name, exprs in name_to_xorq_exprs.items()
    }
    xo_models = {
        name: exprs["fitted_pipe"].fitted_steps[-1].model
        for name, exprs in name_to_xorq_exprs.items()
    }

    for name, acc in xo_accuracies.items():
        print(f"  xorq   {name}: Accuracy = {acc:.3f}")

    return xo_accuracies, xo_models


# =========================================================================
# SKLEARN WAY -- eager
# =========================================================================


def _fit_sklearn_pipeline(name, pipeline, X, y):
    fitted = clone(pipeline).fit(X, y)
    acc = accuracy_score(y, fitted.predict(X))
    print(f"  sklearn {name}: Accuracy = {acc:.3f}")
    return {"accuracy": acc, "model": fitted[-1]}


def compare_result(name, sklearn_result, xorq_result):
    sk_acc, xo_acc = (dct["metrics"]["acc"] for dct in (sklearn_result, xorq_result))
    print(
        f"{name} Accuracy - sklearn: {sk_acc:.3f}, xorq: {xo_acc:.3f}, diff: {abs(sk_acc - xo_acc):.4f}"
    )
    np.testing.assert_allclose(sk_acc, xo_acc, rtol=1e-2)


def compare_results(comparator):
    assert sorted(sklearn_results := comparator.sklearn_results) == sorted(
        xorq_results := comparator.xorq_results
    )
    print("\n=== Comparing Results ===")
    for name, sklearn_result in sklearn_results.items():
        xorq_result = xorq_results[name]
        compare_result(name, sklearn_result, xorq_result)
    print("Assertions passed: sklearn and xorq accuracies match.")


def split_data(df):
    return (df, df)


methods = (MULTINOMIAL, OVR) = ("multinomial", "ovr")
names_pipelines = (
    (
        MULTINOMIAL,
        SklearnPipeline(
            [("multinomial", LogisticRegression(random_state=RANDOM_STATE))]
        ),
    ),
    (
        OVR,
        SklearnPipeline(
            [
                (
                    "ovr",
                    OneVsRestClassifier(LogisticRegression(random_state=RANDOM_STATE)),
                )
            ]
        ),
    ),
)
# con = xo.connect()
# data = con.register(_load_data(), "blobs")
#
# # One-vs-Rest requires a separate connection to avoid backend conflicts
# con_ovr = xo.connect()
# data_ovr = con_ovr.register(_load_data(), "blobs_ovr")
metrics_names_funcs = (("acc", accuracy_score),)
comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=TARGET_COL,
    pred=PRED_COL,
    metrics_names_funcs=metrics_names_funcs,
    load_data=load_data,
    split_data=split_data,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)
(xorq_multinomial_preds, xorq_ovr_preds) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    comparator.result_comparison
    save_comparison_plots(comparator)


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
