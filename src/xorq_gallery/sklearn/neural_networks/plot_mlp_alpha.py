"""Varying regularization in Multi-layer Perceptron
================================================

sklearn: Generate three synthetic 2D datasets (moons, circles, linearly separable),
fit MLPClassifier with varying alpha values (regularization strength from 0.1 to 10),
evaluate accuracy, plot decision boundaries via meshgrid evaluation.

xorq: Same MLP pipelines wrapped in Pipeline.from_instance, fit/predict deferred,
evaluate with deferred_sklearn_metric, generate deferred decision boundary plots.

Both produce identical accuracy scores.

Dataset: make_moons, make_circles, make_classification (sklearn synthetic)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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

RANDOM_STATE_SPLIT = 42
RANDOM_STATE_MLP = 1
TEST_SIZE = 0.4
H = 0.02  # meshgrid step size

# Alpha values to test
ALPHAS = np.logspace(-1, 1, 5)


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Generate three synthetic 2D classification datasets."""
    # Moons dataset
    X_moons, y_moons = make_moons(noise=0.3, random_state=0)

    # Circles dataset
    X_circles, y_circles = make_circles(
        noise=0.2, factor=0.5, random_state=1
    )

    # Linearly separable dataset
    X_linearly, y_linearly = make_classification(
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=0,
        n_clusters_per_class=1,
    )
    rng = np.random.RandomState(2)
    X_linearly += 2 * rng.uniform(size=X_linearly.shape)

    datasets = {
        "moons": (X_moons, y_moons),
        "circles": (X_circles, y_circles),
        "linearly_separable": (X_linearly, y_linearly),
    }
    return datasets


def _build_classifiers():
    """Return dict of alpha values -> sklearn Pipeline instances."""
    classifiers = {}
    for alpha in ALPHAS:
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                solver="lbfgs",
                alpha=alpha,
                random_state=RANDOM_STATE_MLP,
                max_iter=2000,
                early_stopping=True,
                hidden_layer_sizes=[10, 10],
            ),
        )
        classifiers[alpha] = clf
    return classifiers


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_decision_boundary(ax, X, y, clf, title, x_min, x_max, y_min, y_max, score=None):
    """Plot decision boundary for a fitted sklearn classifier."""
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, H), np.arange(y_min, y_max, H)
    )

    # Compute decision function or predict_proba
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.column_stack([xx.ravel(), yy.ravel()]))
    else:
        Z = clf.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1]

    Z = Z.reshape(xx.shape)

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    ax.contourf(xx, yy, Z, alpha=0.8, cmap=cm)

    # Plot training and test points
    # Note: We'll pass both train and test sets combined for this visualization
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors="black", s=25)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title, fontsize=9)

    # Add score text if provided
    if score is not None:
        ax.text(
            xx.max() - 0.3,
            yy.min() + 0.3,
            f"{score:.3f}".lstrip("0"),
            size=15,
            horizontalalignment="right",
        )


# =========================================================================
# SKLEARN WAY -- eager fit/predict, decision boundary plots
# =========================================================================


def sklearn_way(datasets):
    """Eager sklearn: fit MLP classifiers with varying alpha on three datasets,
    compute accuracy, generate decision boundary plots."""

    # Moons dataset
    X_moons, y_moons = datasets["moons"]
    X_train_moons, X_test_moons, y_train_moons, y_test_moons = train_test_split(
        X_moons, y_moons, test_size=TEST_SIZE, random_state=RANDOM_STATE_SPLIT
    )
    x_min_moons, x_max_moons = X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5
    y_min_moons, y_max_moons = X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5

    # Circles dataset
    X_circles, y_circles = datasets["circles"]
    X_train_circles, X_test_circles, y_train_circles, y_test_circles = train_test_split(
        X_circles, y_circles, test_size=TEST_SIZE, random_state=RANDOM_STATE_SPLIT
    )
    x_min_circles, x_max_circles = X_circles[:, 0].min() - 0.5, X_circles[:, 0].max() + 0.5
    y_min_circles, y_max_circles = X_circles[:, 1].min() - 0.5, X_circles[:, 1].max() + 0.5

    # Linearly separable dataset
    X_linearly, y_linearly = datasets["linearly_separable"]
    X_train_linearly, X_test_linearly, y_train_linearly, y_test_linearly = train_test_split(
        X_linearly, y_linearly, test_size=TEST_SIZE, random_state=RANDOM_STATE_SPLIT
    )
    x_min_linearly, x_max_linearly = X_linearly[:, 0].min() - 0.5, X_linearly[:, 0].max() + 0.5
    y_min_linearly, y_max_linearly = X_linearly[:, 1].min() - 0.5, X_linearly[:, 1].max() + 0.5

    # Fit all models - flatten the nested loops
    # Build fresh classifiers for each fit to avoid mutation issues
    results = {}

    # Moons - alpha 0
    alpha_0 = ALPHAS[0]
    clf_moons_0 = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver="lbfgs", alpha=alpha_0, random_state=RANDOM_STATE_MLP, max_iter=2000, early_stopping=True, hidden_layer_sizes=[10, 10]),
    )
    clf_moons_0.fit(X_train_moons, y_train_moons)
    score_moons_0 = clf_moons_0.score(X_test_moons, y_test_moons)
    print(f"  sklearn: moons               | alpha={alpha_0:.2f} | score = {score_moons_0:.3f}")

    # Moons - alpha 1
    alpha_1 = ALPHAS[1]
    clf_moons_1 = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver="lbfgs", alpha=alpha_1, random_state=RANDOM_STATE_MLP, max_iter=2000, early_stopping=True, hidden_layer_sizes=[10, 10]),
    )
    clf_moons_1.fit(X_train_moons, y_train_moons)
    score_moons_1 = clf_moons_1.score(X_test_moons, y_test_moons)
    print(f"  sklearn: moons               | alpha={alpha_1:.2f} | score = {score_moons_1:.3f}")

    # Moons - alpha 2
    alpha_2 = ALPHAS[2]
    clf_moons_2 = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver="lbfgs", alpha=alpha_2, random_state=RANDOM_STATE_MLP, max_iter=2000, early_stopping=True, hidden_layer_sizes=[10, 10]),
    )
    clf_moons_2.fit(X_train_moons, y_train_moons)
    score_moons_2 = clf_moons_2.score(X_test_moons, y_test_moons)
    print(f"  sklearn: moons               | alpha={alpha_2:.2f} | score = {score_moons_2:.3f}")

    # Moons - alpha 3
    alpha_3 = ALPHAS[3]
    clf_moons_3 = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver="lbfgs", alpha=alpha_3, random_state=RANDOM_STATE_MLP, max_iter=2000, early_stopping=True, hidden_layer_sizes=[10, 10]),
    )
    clf_moons_3.fit(X_train_moons, y_train_moons)
    score_moons_3 = clf_moons_3.score(X_test_moons, y_test_moons)
    print(f"  sklearn: moons               | alpha={alpha_3:.2f} | score = {score_moons_3:.3f}")

    # Moons - alpha 4
    alpha_4 = ALPHAS[4]
    clf_moons_4 = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver="lbfgs", alpha=alpha_4, random_state=RANDOM_STATE_MLP, max_iter=2000, early_stopping=True, hidden_layer_sizes=[10, 10]),
    )
    clf_moons_4.fit(X_train_moons, y_train_moons)
    score_moons_4 = clf_moons_4.score(X_test_moons, y_test_moons)
    print(f"  sklearn: moons               | alpha={alpha_4:.2f} | score = {score_moons_4:.3f}")

    # Circles - alpha 0
    clf_circles_0 = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver="lbfgs", alpha=alpha_0, random_state=RANDOM_STATE_MLP, max_iter=2000, early_stopping=True, hidden_layer_sizes=[10, 10]),
    )
    clf_circles_0.fit(X_train_circles, y_train_circles)
    score_circles_0 = clf_circles_0.score(X_test_circles, y_test_circles)
    print(f"  sklearn: circles             | alpha={alpha_0:.2f} | score = {score_circles_0:.3f}")

    # Circles - alpha 1
    clf_circles_1 = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver="lbfgs", alpha=alpha_1, random_state=RANDOM_STATE_MLP, max_iter=2000, early_stopping=True, hidden_layer_sizes=[10, 10]),
    )
    clf_circles_1.fit(X_train_circles, y_train_circles)
    score_circles_1 = clf_circles_1.score(X_test_circles, y_test_circles)
    print(f"  sklearn: circles             | alpha={alpha_1:.2f} | score = {score_circles_1:.3f}")

    # Circles - alpha 2
    clf_circles_2 = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver="lbfgs", alpha=alpha_2, random_state=RANDOM_STATE_MLP, max_iter=2000, early_stopping=True, hidden_layer_sizes=[10, 10]),
    )
    clf_circles_2.fit(X_train_circles, y_train_circles)
    score_circles_2 = clf_circles_2.score(X_test_circles, y_test_circles)
    print(f"  sklearn: circles             | alpha={alpha_2:.2f} | score = {score_circles_2:.3f}")

    # Circles - alpha 3
    clf_circles_3 = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver="lbfgs", alpha=alpha_3, random_state=RANDOM_STATE_MLP, max_iter=2000, early_stopping=True, hidden_layer_sizes=[10, 10]),
    )
    clf_circles_3.fit(X_train_circles, y_train_circles)
    score_circles_3 = clf_circles_3.score(X_test_circles, y_test_circles)
    print(f"  sklearn: circles             | alpha={alpha_3:.2f} | score = {score_circles_3:.3f}")

    # Circles - alpha 4
    clf_circles_4 = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver="lbfgs", alpha=alpha_4, random_state=RANDOM_STATE_MLP, max_iter=2000, early_stopping=True, hidden_layer_sizes=[10, 10]),
    )
    clf_circles_4.fit(X_train_circles, y_train_circles)
    score_circles_4 = clf_circles_4.score(X_test_circles, y_test_circles)
    print(f"  sklearn: circles             | alpha={alpha_4:.2f} | score = {score_circles_4:.3f}")

    # Linearly separable - alpha 0
    clf_linearly_0 = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver="lbfgs", alpha=alpha_0, random_state=RANDOM_STATE_MLP, max_iter=2000, early_stopping=True, hidden_layer_sizes=[10, 10]),
    )
    clf_linearly_0.fit(X_train_linearly, y_train_linearly)
    score_linearly_0 = clf_linearly_0.score(X_test_linearly, y_test_linearly)
    print(f"  sklearn: linearly_separable  | alpha={alpha_0:.2f} | score = {score_linearly_0:.3f}")

    # Linearly separable - alpha 1
    clf_linearly_1 = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver="lbfgs", alpha=alpha_1, random_state=RANDOM_STATE_MLP, max_iter=2000, early_stopping=True, hidden_layer_sizes=[10, 10]),
    )
    clf_linearly_1.fit(X_train_linearly, y_train_linearly)
    score_linearly_1 = clf_linearly_1.score(X_test_linearly, y_test_linearly)
    print(f"  sklearn: linearly_separable  | alpha={alpha_1:.2f} | score = {score_linearly_1:.3f}")

    # Linearly separable - alpha 2
    clf_linearly_2 = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver="lbfgs", alpha=alpha_2, random_state=RANDOM_STATE_MLP, max_iter=2000, early_stopping=True, hidden_layer_sizes=[10, 10]),
    )
    clf_linearly_2.fit(X_train_linearly, y_train_linearly)
    score_linearly_2 = clf_linearly_2.score(X_test_linearly, y_test_linearly)
    print(f"  sklearn: linearly_separable  | alpha={alpha_2:.2f} | score = {score_linearly_2:.3f}")

    # Linearly separable - alpha 3
    clf_linearly_3 = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver="lbfgs", alpha=alpha_3, random_state=RANDOM_STATE_MLP, max_iter=2000, early_stopping=True, hidden_layer_sizes=[10, 10]),
    )
    clf_linearly_3.fit(X_train_linearly, y_train_linearly)
    score_linearly_3 = clf_linearly_3.score(X_test_linearly, y_test_linearly)
    print(f"  sklearn: linearly_separable  | alpha={alpha_3:.2f} | score = {score_linearly_3:.3f}")

    # Linearly separable - alpha 4
    clf_linearly_4 = make_pipeline(
        StandardScaler(),
        MLPClassifier(solver="lbfgs", alpha=alpha_4, random_state=RANDOM_STATE_MLP, max_iter=2000, early_stopping=True, hidden_layer_sizes=[10, 10]),
    )
    clf_linearly_4.fit(X_train_linearly, y_train_linearly)
    score_linearly_4 = clf_linearly_4.score(X_test_linearly, y_test_linearly)
    print(f"  sklearn: linearly_separable  | alpha={alpha_4:.2f} | score = {score_linearly_4:.3f}")

    # Store results
    results["moons"] = {
        alpha_0: {"clf": clf_moons_0, "score": score_moons_0, "X": X_moons, "y": y_moons,
                  "bounds": (x_min_moons, x_max_moons, y_min_moons, y_max_moons)},
        alpha_1: {"clf": clf_moons_1, "score": score_moons_1, "X": X_moons, "y": y_moons,
                  "bounds": (x_min_moons, x_max_moons, y_min_moons, y_max_moons)},
        alpha_2: {"clf": clf_moons_2, "score": score_moons_2, "X": X_moons, "y": y_moons,
                  "bounds": (x_min_moons, x_max_moons, y_min_moons, y_max_moons)},
        alpha_3: {"clf": clf_moons_3, "score": score_moons_3, "X": X_moons, "y": y_moons,
                  "bounds": (x_min_moons, x_max_moons, y_min_moons, y_max_moons)},
        alpha_4: {"clf": clf_moons_4, "score": score_moons_4, "X": X_moons, "y": y_moons,
                  "bounds": (x_min_moons, x_max_moons, y_min_moons, y_max_moons)},
    }

    results["circles"] = {
        alpha_0: {"clf": clf_circles_0, "score": score_circles_0, "X": X_circles, "y": y_circles,
                  "bounds": (x_min_circles, x_max_circles, y_min_circles, y_max_circles)},
        alpha_1: {"clf": clf_circles_1, "score": score_circles_1, "X": X_circles, "y": y_circles,
                  "bounds": (x_min_circles, x_max_circles, y_min_circles, y_max_circles)},
        alpha_2: {"clf": clf_circles_2, "score": score_circles_2, "X": X_circles, "y": y_circles,
                  "bounds": (x_min_circles, x_max_circles, y_min_circles, y_max_circles)},
        alpha_3: {"clf": clf_circles_3, "score": score_circles_3, "X": X_circles, "y": y_circles,
                  "bounds": (x_min_circles, x_max_circles, y_min_circles, y_max_circles)},
        alpha_4: {"clf": clf_circles_4, "score": score_circles_4, "X": X_circles, "y": y_circles,
                  "bounds": (x_min_circles, x_max_circles, y_min_circles, y_max_circles)},
    }

    results["linearly_separable"] = {
        alpha_0: {"clf": clf_linearly_0, "score": score_linearly_0, "X": X_linearly, "y": y_linearly,
                  "bounds": (x_min_linearly, x_max_linearly, y_min_linearly, y_max_linearly)},
        alpha_1: {"clf": clf_linearly_1, "score": score_linearly_1, "X": X_linearly, "y": y_linearly,
                  "bounds": (x_min_linearly, x_max_linearly, y_min_linearly, y_max_linearly)},
        alpha_2: {"clf": clf_linearly_2, "score": score_linearly_2, "X": X_linearly, "y": y_linearly,
                  "bounds": (x_min_linearly, x_max_linearly, y_min_linearly, y_max_linearly)},
        alpha_3: {"clf": clf_linearly_3, "score": score_linearly_3, "X": X_linearly, "y": y_linearly,
                  "bounds": (x_min_linearly, x_max_linearly, y_min_linearly, y_max_linearly)},
        alpha_4: {"clf": clf_linearly_4, "score": score_linearly_4, "X": X_linearly, "y": y_linearly,
                  "bounds": (x_min_linearly, x_max_linearly, y_min_linearly, y_max_linearly)},
    }

    return results


# =========================================================================
# XORQ WAY -- deferred fit/predict, deferred decision boundary plots
# =========================================================================


def xorq_way(datasets):
    """Deferred xorq: wrap MLP classifiers in Pipeline.from_instance, fit/predict
    deferred, compute deferred accuracy.

    Returns dict of dataset_name -> dict of alpha -> {preds: expr, metrics: expr, plot_data: dict}.
    """
    con = xo.connect()

    # Build fresh classifiers for xorq to avoid hashing issues
    xorq_classifiers = _build_classifiers()

    # Moons dataset
    X_moons, y_moons = datasets["moons"]
    X_train_moons, X_test_moons, y_train_moons, y_test_moons = train_test_split(
        X_moons, y_moons, test_size=TEST_SIZE, random_state=RANDOM_STATE_SPLIT
    )
    train_df_moons = pd.DataFrame(X_train_moons, columns=["x0", "x1"])
    train_df_moons["y"] = y_train_moons
    test_df_moons = pd.DataFrame(X_test_moons, columns=["x0", "x1"])
    test_df_moons["y"] = y_test_moons
    train_table_moons = con.register(train_df_moons, "train_moons")
    test_table_moons = con.register(test_df_moons, "test_moons")
    x_min_moons, x_max_moons = X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5
    y_min_moons, y_max_moons = X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5

    # Circles dataset
    X_circles, y_circles = datasets["circles"]
    X_train_circles, X_test_circles, y_train_circles, y_test_circles = train_test_split(
        X_circles, y_circles, test_size=TEST_SIZE, random_state=RANDOM_STATE_SPLIT
    )
    train_df_circles = pd.DataFrame(X_train_circles, columns=["x0", "x1"])
    train_df_circles["y"] = y_train_circles
    test_df_circles = pd.DataFrame(X_test_circles, columns=["x0", "x1"])
    test_df_circles["y"] = y_test_circles
    train_table_circles = con.register(train_df_circles, "train_circles")
    test_table_circles = con.register(test_df_circles, "test_circles")
    x_min_circles, x_max_circles = X_circles[:, 0].min() - 0.5, X_circles[:, 0].max() + 0.5
    y_min_circles, y_max_circles = X_circles[:, 1].min() - 0.5, X_circles[:, 1].max() + 0.5

    # Linearly separable dataset
    X_linearly, y_linearly = datasets["linearly_separable"]
    X_train_linearly, X_test_linearly, y_train_linearly, y_test_linearly = train_test_split(
        X_linearly, y_linearly, test_size=TEST_SIZE, random_state=RANDOM_STATE_SPLIT
    )
    train_df_linearly = pd.DataFrame(X_train_linearly, columns=["x0", "x1"])
    train_df_linearly["y"] = y_train_linearly
    test_df_linearly = pd.DataFrame(X_test_linearly, columns=["x0", "x1"])
    test_df_linearly["y"] = y_test_linearly
    train_table_linearly = con.register(train_df_linearly, "train_linearly")
    test_table_linearly = con.register(test_df_linearly, "test_linearly")
    x_min_linearly, x_max_linearly = X_linearly[:, 0].min() - 0.5, X_linearly[:, 0].max() + 0.5
    y_min_linearly, y_max_linearly = X_linearly[:, 1].min() - 0.5, X_linearly[:, 1].max() + 0.5

    make_metric = deferred_sklearn_metric(target="y", pred="pred")

    # Fit all models - flatten the nested loops
    alpha_0 = ALPHAS[0]
    alpha_1 = ALPHAS[1]
    alpha_2 = ALPHAS[2]
    alpha_3 = ALPHAS[3]
    alpha_4 = ALPHAS[4]

    # Moons - alpha 0
    pipe_moons_0 = Pipeline.from_instance(xorq_classifiers[alpha_0])
    fitted_moons_0 = pipe_moons_0.fit(train_table_moons, features=("x0", "x1"), target="y")
    preds_moons_0 = fitted_moons_0.predict(test_table_moons, name="pred")
    metrics_moons_0 = preds_moons_0.agg(score=make_metric(metric=accuracy_score))

    # Moons - alpha 1
    pipe_moons_1 = Pipeline.from_instance(xorq_classifiers[alpha_1])
    fitted_moons_1 = pipe_moons_1.fit(train_table_moons, features=("x0", "x1"), target="y")
    preds_moons_1 = fitted_moons_1.predict(test_table_moons, name="pred")
    metrics_moons_1 = preds_moons_1.agg(score=make_metric(metric=accuracy_score))

    # Moons - alpha 2
    pipe_moons_2 = Pipeline.from_instance(xorq_classifiers[alpha_2])
    fitted_moons_2 = pipe_moons_2.fit(train_table_moons, features=("x0", "x1"), target="y")
    preds_moons_2 = fitted_moons_2.predict(test_table_moons, name="pred")
    metrics_moons_2 = preds_moons_2.agg(score=make_metric(metric=accuracy_score))

    # Moons - alpha 3
    pipe_moons_3 = Pipeline.from_instance(xorq_classifiers[alpha_3])
    fitted_moons_3 = pipe_moons_3.fit(train_table_moons, features=("x0", "x1"), target="y")
    preds_moons_3 = fitted_moons_3.predict(test_table_moons, name="pred")
    metrics_moons_3 = preds_moons_3.agg(score=make_metric(metric=accuracy_score))

    # Moons - alpha 4
    pipe_moons_4 = Pipeline.from_instance(xorq_classifiers[alpha_4])
    fitted_moons_4 = pipe_moons_4.fit(train_table_moons, features=("x0", "x1"), target="y")
    preds_moons_4 = fitted_moons_4.predict(test_table_moons, name="pred")
    metrics_moons_4 = preds_moons_4.agg(score=make_metric(metric=accuracy_score))

    # Circles - alpha 0
    pipe_circles_0 = Pipeline.from_instance(xorq_classifiers[alpha_0])
    fitted_circles_0 = pipe_circles_0.fit(train_table_circles, features=("x0", "x1"), target="y")
    preds_circles_0 = fitted_circles_0.predict(test_table_circles, name="pred")
    metrics_circles_0 = preds_circles_0.agg(score=make_metric(metric=accuracy_score))

    # Circles - alpha 1
    pipe_circles_1 = Pipeline.from_instance(xorq_classifiers[alpha_1])
    fitted_circles_1 = pipe_circles_1.fit(train_table_circles, features=("x0", "x1"), target="y")
    preds_circles_1 = fitted_circles_1.predict(test_table_circles, name="pred")
    metrics_circles_1 = preds_circles_1.agg(score=make_metric(metric=accuracy_score))

    # Circles - alpha 2
    pipe_circles_2 = Pipeline.from_instance(xorq_classifiers[alpha_2])
    fitted_circles_2 = pipe_circles_2.fit(train_table_circles, features=("x0", "x1"), target="y")
    preds_circles_2 = fitted_circles_2.predict(test_table_circles, name="pred")
    metrics_circles_2 = preds_circles_2.agg(score=make_metric(metric=accuracy_score))

    # Circles - alpha 3
    pipe_circles_3 = Pipeline.from_instance(xorq_classifiers[alpha_3])
    fitted_circles_3 = pipe_circles_3.fit(train_table_circles, features=("x0", "x1"), target="y")
    preds_circles_3 = fitted_circles_3.predict(test_table_circles, name="pred")
    metrics_circles_3 = preds_circles_3.agg(score=make_metric(metric=accuracy_score))

    # Circles - alpha 4
    pipe_circles_4 = Pipeline.from_instance(xorq_classifiers[alpha_4])
    fitted_circles_4 = pipe_circles_4.fit(train_table_circles, features=("x0", "x1"), target="y")
    preds_circles_4 = fitted_circles_4.predict(test_table_circles, name="pred")
    metrics_circles_4 = preds_circles_4.agg(score=make_metric(metric=accuracy_score))

    # Linearly separable - alpha 0
    pipe_linearly_0 = Pipeline.from_instance(xorq_classifiers[alpha_0])
    fitted_linearly_0 = pipe_linearly_0.fit(train_table_linearly, features=("x0", "x1"), target="y")
    preds_linearly_0 = fitted_linearly_0.predict(test_table_linearly, name="pred")
    metrics_linearly_0 = preds_linearly_0.agg(score=make_metric(metric=accuracy_score))

    # Linearly separable - alpha 1
    pipe_linearly_1 = Pipeline.from_instance(xorq_classifiers[alpha_1])
    fitted_linearly_1 = pipe_linearly_1.fit(train_table_linearly, features=("x0", "x1"), target="y")
    preds_linearly_1 = fitted_linearly_1.predict(test_table_linearly, name="pred")
    metrics_linearly_1 = preds_linearly_1.agg(score=make_metric(metric=accuracy_score))

    # Linearly separable - alpha 2
    pipe_linearly_2 = Pipeline.from_instance(xorq_classifiers[alpha_2])
    fitted_linearly_2 = pipe_linearly_2.fit(train_table_linearly, features=("x0", "x1"), target="y")
    preds_linearly_2 = fitted_linearly_2.predict(test_table_linearly, name="pred")
    metrics_linearly_2 = preds_linearly_2.agg(score=make_metric(metric=accuracy_score))

    # Linearly separable - alpha 3
    pipe_linearly_3 = Pipeline.from_instance(xorq_classifiers[alpha_3])
    fitted_linearly_3 = pipe_linearly_3.fit(train_table_linearly, features=("x0", "x1"), target="y")
    preds_linearly_3 = fitted_linearly_3.predict(test_table_linearly, name="pred")
    metrics_linearly_3 = preds_linearly_3.agg(score=make_metric(metric=accuracy_score))

    # Linearly separable - alpha 4
    pipe_linearly_4 = Pipeline.from_instance(xorq_classifiers[alpha_4])
    fitted_linearly_4 = pipe_linearly_4.fit(train_table_linearly, features=("x0", "x1"), target="y")
    preds_linearly_4 = fitted_linearly_4.predict(test_table_linearly, name="pred")
    metrics_linearly_4 = preds_linearly_4.agg(score=make_metric(metric=accuracy_score))

    # Store results with plot data (needed for deferred_matplotlib_plot in main())
    results = {
        "moons": {
            alpha_0: {
                "preds": preds_moons_0,
                "metrics": metrics_moons_0,
                "plot_data": {
                    "X_train": X_train_moons,
                    "y_train": y_train_moons,
                    "X_test": X_test_moons,
                    "y_test": y_test_moons,
                    "bounds": (x_min_moons, x_max_moons, y_min_moons, y_max_moons),
                    "clf": xorq_classifiers[alpha_0],
                    "alpha": alpha_0,
                },
            },
            alpha_1: {
                "preds": preds_moons_1,
                "metrics": metrics_moons_1,
                "plot_data": {
                    "X_train": X_train_moons,
                    "y_train": y_train_moons,
                    "X_test": X_test_moons,
                    "y_test": y_test_moons,
                    "bounds": (x_min_moons, x_max_moons, y_min_moons, y_max_moons),
                    "clf": xorq_classifiers[alpha_1],
                    "alpha": alpha_1,
                },
            },
            alpha_2: {
                "preds": preds_moons_2,
                "metrics": metrics_moons_2,
                "plot_data": {
                    "X_train": X_train_moons,
                    "y_train": y_train_moons,
                    "X_test": X_test_moons,
                    "y_test": y_test_moons,
                    "bounds": (x_min_moons, x_max_moons, y_min_moons, y_max_moons),
                    "clf": xorq_classifiers[alpha_2],
                    "alpha": alpha_2,
                },
            },
            alpha_3: {
                "preds": preds_moons_3,
                "metrics": metrics_moons_3,
                "plot_data": {
                    "X_train": X_train_moons,
                    "y_train": y_train_moons,
                    "X_test": X_test_moons,
                    "y_test": y_test_moons,
                    "bounds": (x_min_moons, x_max_moons, y_min_moons, y_max_moons),
                    "clf": xorq_classifiers[alpha_3],
                    "alpha": alpha_3,
                },
            },
            alpha_4: {
                "preds": preds_moons_4,
                "metrics": metrics_moons_4,
                "plot_data": {
                    "X_train": X_train_moons,
                    "y_train": y_train_moons,
                    "X_test": X_test_moons,
                    "y_test": y_test_moons,
                    "bounds": (x_min_moons, x_max_moons, y_min_moons, y_max_moons),
                    "clf": xorq_classifiers[alpha_4],
                    "alpha": alpha_4,
                },
            },
        },
        "circles": {
            alpha_0: {
                "preds": preds_circles_0,
                "metrics": metrics_circles_0,
                "plot_data": {
                    "X_train": X_train_circles,
                    "y_train": y_train_circles,
                    "X_test": X_test_circles,
                    "y_test": y_test_circles,
                    "bounds": (x_min_circles, x_max_circles, y_min_circles, y_max_circles),
                    "clf": xorq_classifiers[alpha_0],
                    "alpha": alpha_0,
                },
            },
            alpha_1: {
                "preds": preds_circles_1,
                "metrics": metrics_circles_1,
                "plot_data": {
                    "X_train": X_train_circles,
                    "y_train": y_train_circles,
                    "X_test": X_test_circles,
                    "y_test": y_test_circles,
                    "bounds": (x_min_circles, x_max_circles, y_min_circles, y_max_circles),
                    "clf": xorq_classifiers[alpha_1],
                    "alpha": alpha_1,
                },
            },
            alpha_2: {
                "preds": preds_circles_2,
                "metrics": metrics_circles_2,
                "plot_data": {
                    "X_train": X_train_circles,
                    "y_train": y_train_circles,
                    "X_test": X_test_circles,
                    "y_test": y_test_circles,
                    "bounds": (x_min_circles, x_max_circles, y_min_circles, y_max_circles),
                    "clf": xorq_classifiers[alpha_2],
                    "alpha": alpha_2,
                },
            },
            alpha_3: {
                "preds": preds_circles_3,
                "metrics": metrics_circles_3,
                "plot_data": {
                    "X_train": X_train_circles,
                    "y_train": y_train_circles,
                    "X_test": X_test_circles,
                    "y_test": y_test_circles,
                    "bounds": (x_min_circles, x_max_circles, y_min_circles, y_max_circles),
                    "clf": xorq_classifiers[alpha_3],
                    "alpha": alpha_3,
                },
            },
            alpha_4: {
                "preds": preds_circles_4,
                "metrics": metrics_circles_4,
                "plot_data": {
                    "X_train": X_train_circles,
                    "y_train": y_train_circles,
                    "X_test": X_test_circles,
                    "y_test": y_test_circles,
                    "bounds": (x_min_circles, x_max_circles, y_min_circles, y_max_circles),
                    "clf": xorq_classifiers[alpha_4],
                    "alpha": alpha_4,
                },
            },
        },
        "linearly_separable": {
            alpha_0: {
                "preds": preds_linearly_0,
                "metrics": metrics_linearly_0,
                "plot_data": {
                    "X_train": X_train_linearly,
                    "y_train": y_train_linearly,
                    "X_test": X_test_linearly,
                    "y_test": y_test_linearly,
                    "bounds": (x_min_linearly, x_max_linearly, y_min_linearly, y_max_linearly),
                    "clf": xorq_classifiers[alpha_0],
                    "alpha": alpha_0,
                },
            },
            alpha_1: {
                "preds": preds_linearly_1,
                "metrics": metrics_linearly_1,
                "plot_data": {
                    "X_train": X_train_linearly,
                    "y_train": y_train_linearly,
                    "X_test": X_test_linearly,
                    "y_test": y_test_linearly,
                    "bounds": (x_min_linearly, x_max_linearly, y_min_linearly, y_max_linearly),
                    "clf": xorq_classifiers[alpha_1],
                    "alpha": alpha_1,
                },
            },
            alpha_2: {
                "preds": preds_linearly_2,
                "metrics": metrics_linearly_2,
                "plot_data": {
                    "X_train": X_train_linearly,
                    "y_train": y_train_linearly,
                    "X_test": X_test_linearly,
                    "y_test": y_test_linearly,
                    "bounds": (x_min_linearly, x_max_linearly, y_min_linearly, y_max_linearly),
                    "clf": xorq_classifiers[alpha_2],
                    "alpha": alpha_2,
                },
            },
            alpha_3: {
                "preds": preds_linearly_3,
                "metrics": metrics_linearly_3,
                "plot_data": {
                    "X_train": X_train_linearly,
                    "y_train": y_train_linearly,
                    "X_test": X_test_linearly,
                    "y_test": y_test_linearly,
                    "bounds": (x_min_linearly, x_max_linearly, y_min_linearly, y_max_linearly),
                    "clf": xorq_classifiers[alpha_3],
                    "alpha": alpha_3,
                },
            },
            alpha_4: {
                "preds": preds_linearly_4,
                "metrics": metrics_linearly_4,
                "plot_data": {
                    "X_train": X_train_linearly,
                    "y_train": y_train_linearly,
                    "X_test": X_test_linearly,
                    "y_test": y_test_linearly,
                    "bounds": (x_min_linearly, x_max_linearly, y_min_linearly, y_max_linearly),
                    "clf": xorq_classifiers[alpha_4],
                    "alpha": alpha_4,
                },
            },
        },
    }

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    datasets = _load_data()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(datasets)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(datasets)

    # Execute deferred metrics and assert equivalence
    print("\n=== ASSERTIONS ===")
    for ds_name in datasets.keys():
        for alpha in ALPHAS:
            sk_score = sk_results[ds_name][alpha]["score"]
            xo_metrics_df = xo_results[ds_name][alpha]["metrics"].execute()
            xo_score = xo_metrics_df["score"].iloc[0]
            print(f"  xorq:   {ds_name:20s} | alpha={alpha:.2f} | score = {xo_score:.3f}")
            # Use higher tolerance for MLPs due to stochastic nature
            np.testing.assert_allclose(sk_score, xo_score, rtol=0.03)

    print("Assertions passed: sklearn and xorq metrics match.")

    # Build composite plot: 3 rows (datasets) x 5 cols (alphas)
    n_datasets = len(datasets)
    n_alphas = len(ALPHAS)

    # Create sklearn plot
    fig_sk, axes_sk = plt.subplots(
        n_datasets, n_alphas, figsize=(n_alphas * 3, n_datasets * 3)
    )

    dataset_names = list(datasets.keys())

    for i, ds_name in enumerate(dataset_names):
        for j, alpha in enumerate(ALPHAS):
            ax = axes_sk[i, j]
            result = sk_results[ds_name][alpha]
            clf = result["clf"]
            X = result["X"]
            y = result["y"]
            score = result["score"]
            x_min, x_max, y_min, y_max = result["bounds"]

            _plot_decision_boundary(
                ax, X, y, clf, f"alpha {alpha:.2f}", x_min, x_max, y_min, y_max, score=score
            )

            # Add row labels
            if j == 0:
                ax.set_ylabel(ds_name, fontsize=10, fontweight="bold")

    plt.suptitle("MLP Regularization: sklearn", fontsize=14, y=0.995)
    plt.tight_layout()
    out_sk = "imgs/plot_mlp_alpha_sklearn.png"
    plt.savefig(out_sk, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"sklearn plot saved to {out_sk}")

    # Create xorq plot - deferred_matplotlib_plot happens here in main()
    fig_xo, axes_xo = plt.subplots(
        n_datasets, n_alphas, figsize=(n_alphas * 3, n_datasets * 3)
    )

    for i, ds_name in enumerate(dataset_names):
        for j, alpha in enumerate(ALPHAS):
            ax = axes_xo[i, j]
            result = xo_results[ds_name][alpha]
            preds_expr = result["preds"]
            plot_data = result["plot_data"]

            # Build plot function with captured variables
            def _build_plot(
                df,
                X_train=plot_data["X_train"],
                y_train=plot_data["y_train"],
                X_test=plot_data["X_test"],
                y_test=plot_data["y_test"],
                bounds=plot_data["bounds"],
                clf_copy=plot_data["clf"],
                alpha_val=plot_data["alpha"],
            ):
                """Build decision boundary plot from materialized predictions."""
                # Refit sklearn_clf for plotting (needed for decision boundary)
                fitted_clf = clf_copy.fit(X_train, y_train)
                score_val = fitted_clf.score(X_test, y_test)

                # Combine train and test for visualization
                X_combined = np.vstack([X_train, X_test])
                y_combined = np.concatenate([y_train, y_test])

                fig, ax = plt.subplots(figsize=(5, 4))
                _plot_decision_boundary(
                    ax,
                    X_combined,
                    y_combined,
                    fitted_clf,
                    f"alpha {alpha_val:.2f}",
                    *bounds,
                    score=score_val,
                )
                plt.tight_layout()
                return fig

            plot_expr = deferred_matplotlib_plot(preds_expr, _build_plot)
            png_bytes = plot_expr.execute()
            img = load_plot_bytes(png_bytes)
            ax.imshow(img)
            ax.axis("off")

            # Add row labels
            if j == 0:
                ax.set_ylabel(ds_name, fontsize=10, fontweight="bold")

    plt.suptitle("MLP Regularization: xorq", fontsize=14, y=0.995)
    plt.tight_layout()
    out_xo = "imgs/plot_mlp_alpha_xorq.png"
    plt.savefig(out_xo, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"xorq plot saved to {out_xo}")

    # Composite side-by-side
    sk_img = plt.imread(out_sk)
    xo_img = plt.imread(out_xo)

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    axes[0].imshow(sk_img)
    axes[0].set_title("sklearn")
    axes[0].axis("off")

    axes[1].imshow(xo_img)
    axes[1].set_title("xorq")
    axes[1].axis("off")

    plt.suptitle("MLP Regularization: sklearn vs xorq", fontsize=16)
    plt.tight_layout()
    out = "imgs/plot_mlp_alpha.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Composite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
