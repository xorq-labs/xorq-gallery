"""Classifier comparison
========================

sklearn: Generate three synthetic 2D datasets (moons, circles, linearly separable),
fit multiple classifiers (KNeighborsClassifier, SVC, DecisionTreeClassifier,
RandomForestClassifier, GaussianNB, MLPClassifier, AdaBoostClassifier) on each,
evaluate accuracy, plot decision boundaries via meshgrid evaluation.

xorq: Same classifiers wrapped in Pipeline.from_instance, fit/predict deferred,
evaluate with deferred_sklearn_metric, generate deferred decision boundary plots.

Both produce identical accuracy scores.

Dataset: make_moons, make_circles, make_classification (sklearn synthetic)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/classification/plot_classifier_comparison.py
"""

from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
)
from xorq_gallery.sklearn.sklearn_lib import (
    make_deferred_xorq_result as _make_deferred_xorq_result,
)
from xorq_gallery.sklearn.sklearn_lib import (
    make_sklearn_result as _make_sklearn_result,
)
from xorq_gallery.utils import fig_to_image, save_fig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_SAMPLES = 100
TEST_SIZE = 0.4
H = 0.02  # meshgrid step size

DATASET_NAMES = ("moons", "circles", "linearly_separable")
CLASSIFIER_NAMES = (
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
)

FEATURE_COLS = ("x0", "x1")
TARGET_COL = "y"
PRED_COL = "pred"


# ---------------------------------------------------------------------------
# Data loading — one load_data function per dataset
# ---------------------------------------------------------------------------


def _generate_dataset(name):
    """Generate a single synthetic 2D classification dataset as DataFrame."""
    if name == "moons":
        X, y = make_moons(n_samples=N_SAMPLES, noise=0.3, random_state=RANDOM_STATE)
    elif name == "circles":
        X, y = make_circles(
            n_samples=N_SAMPLES, noise=0.2, factor=0.5, random_state=RANDOM_STATE
        )
    else:
        X, y = make_classification(
            n_samples=N_SAMPLES,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=RANDOM_STATE,
            n_clusters_per_class=1,
        )
        rng = np.random.RandomState(RANDOM_STATE)
        X += 2 * rng.uniform(size=X.shape)

    return pd.DataFrame(X, columns=list(FEATURE_COLS)).assign(**{TARGET_COL: y})


_LOAD_DATA_FNS = {name: partial(_generate_dataset, name) for name in DATASET_NAMES}


# ---------------------------------------------------------------------------
# Classifiers (shared sklearn Pipeline objects)
# ---------------------------------------------------------------------------

names_pipelines = (
    (
        "Nearest Neighbors",
        SklearnPipeline(
            [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(3))]
        ),
    ),
    (
        "Linear SVM",
        SklearnPipeline(
            [("scaler", StandardScaler()), ("clf", SVC(kernel="linear", C=0.025))]
        ),
    ),
    (
        "RBF SVM",
        SklearnPipeline([("scaler", StandardScaler()), ("clf", SVC(gamma=2, C=1))]),
    ),
    (
        "Decision Tree",
        SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
                ),
            ]
        ),
    ),
    (
        "Random Forest",
        SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        max_depth=5,
                        n_estimators=10,
                        max_features=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    ),
    (
        "Neural Net",
        SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(alpha=1, max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
    ),
    (
        "AdaBoost",
        SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", AdaBoostClassifier(random_state=RANDOM_STATE)),
            ]
        ),
    ),
    (
        "Naive Bayes",
        SklearnPipeline([("scaler", StandardScaler()), ("clf", GaussianNB())]),
    ),
)

metrics_names_funcs = (("accuracy", accuracy_score),)


# ---------------------------------------------------------------------------
# make_other overrides: store full fitted pipeline for decision boundaries
# ---------------------------------------------------------------------------


def _make_sklearn_other(fitted):
    return {"full_pipeline": fitted}


def _make_xorq_other(xorq_fitted):
    steps = [(step.step.name, step.model) for step in xorq_fitted.fitted_steps]
    full_pipeline = SklearnPipeline(steps)
    return {"full_pipeline": lambda: full_pipeline}


make_sklearn_result = _make_sklearn_result(make_other=_make_sklearn_other)
make_deferred_xorq_result = _make_deferred_xorq_result(make_other=_make_xorq_other)


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def compare_results(comparator):
    assert sorted(sklearn_results := comparator.sklearn_results) == sorted(
        xorq_results := comparator.xorq_results
    )
    print("\n=== Comparing Results ===")
    for name, sklearn_result in sklearn_results.items():
        xorq_result = xorq_results[name]
        sk_acc = sklearn_result["metrics"]["accuracy"]
        xo_acc = xorq_result["metrics"]["accuracy"]
        print(f"  {name:20s} accuracy - sklearn: {sk_acc:.4f}, xorq: {xo_acc:.4f}")


def _plot_decision_boundary(ax, X, y, clf, title, x_min, x_max, y_min, y_max):
    """Plot decision boundary for a fitted sklearn classifier."""
    xx, yy = np.meshgrid(np.arange(x_min, x_max, H), np.arange(y_min, y_max, H))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cm = ListedColormap(["#FF9999", "#9999FF"])
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cm)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors="k", s=20)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title, fontsize=9)


def _draw_row(axes, comparator, results_key, label_prefix):
    """Draw a row of decision boundary plots for one framework (sklearn or xorq)."""
    X = comparator.df[list(FEATURE_COLS)].values
    y = comparator.df[TARGET_COL].values
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    results = getattr(comparator, results_key)
    for j, name in enumerate(CLASSIFIER_NAMES):
        acc = results[name]["metrics"]["accuracy"]
        clf = results[name]["other"]["full_pipeline"]
        _plot_decision_boundary(
            axes[j],
            X,
            y,
            clf,
            f"{name}\nacc={acc:.2f}",
            x_min,
            x_max,
            y_min,
            y_max,
        )


def plot_results(comparators):
    """Build composite plot: 3 datasets x 8 classifiers, sklearn top / xorq bottom."""
    n_datasets = len(DATASET_NAMES)
    n_classifiers = len(CLASSIFIER_NAMES)

    # Build sklearn row figure per dataset, then xorq row figure per dataset
    fig_sk, axes_sk = plt.subplots(
        n_datasets, n_classifiers, figsize=(n_classifiers * 2.5, n_datasets * 2.5)
    )
    fig_xo, axes_xo = plt.subplots(
        n_datasets, n_classifiers, figsize=(n_classifiers * 2.5, n_datasets * 2.5)
    )

    for i, ds_name in enumerate(DATASET_NAMES):
        comp = comparators[ds_name]
        _draw_row(axes_sk[i], comp, "sklearn_results", "sklearn")
        _draw_row(axes_xo[i], comp, "xorq_results", "xorq")

        if axes_sk.ndim > 1:
            axes_sk[i, 0].set_ylabel(ds_name, fontsize=10, fontweight="bold")
            axes_xo[i, 0].set_ylabel(ds_name, fontsize=10, fontweight="bold")

    fig_sk.suptitle("sklearn", fontsize=14, fontweight="bold")
    fig_sk.tight_layout()
    fig_xo.suptitle("xorq", fontsize=14, fontweight="bold")
    fig_xo.tight_layout()

    # Composite: sklearn (top) | xorq (bottom)
    fig, axes = plt.subplots(2, 1, figsize=(20, 16))
    axes[0].imshow(fig_to_image(fig_sk))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold", pad=10)
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(fig_xo))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold", pad=10)
    axes[1].axis("off")

    fig.suptitle(
        "Classifier Comparison: sklearn vs xorq", fontsize=16, fontweight="bold"
    )
    fig.tight_layout()

    plt.close(fig_sk)
    plt.close(fig_xo)
    return fig


# ---------------------------------------------------------------------------
# Module-level setup — one comparator per dataset
# ---------------------------------------------------------------------------


# plot_results is called from main() with all comparators, not per-comparator
def _noop_compare(comparator):
    pass


def _noop_plot(comparator):
    return None


comparators = {
    ds_name: SklearnXorqComparator(
        names_pipelines=names_pipelines,
        features=FEATURE_COLS,
        target=TARGET_COL,
        pred=PRED_COL,
        metrics_names_funcs=metrics_names_funcs,
        load_data=_LOAD_DATA_FNS[ds_name],
        split_data=partial(
            train_test_split, test_size=TEST_SIZE, random_state=RANDOM_STATE
        ),
        make_sklearn_result=make_sklearn_result,
        make_deferred_xorq_result=make_deferred_xorq_result,
        compare_results_fn=_noop_compare,
        plot_results_fn=_noop_plot,
    )
    for ds_name in DATASET_NAMES
}

# -- Module-level deferred exprs (for xorq build --expr) -------------------
(
    xorq_moons_nearest_neighbors_preds,
    xorq_moons_linear_svm_preds,
    xorq_moons_rbf_svm_preds,
    xorq_moons_decision_tree_preds,
    xorq_moons_random_forest_preds,
    xorq_moons_neural_net_preds,
    xorq_moons_adaboost_preds,
    xorq_moons_naive_bayes_preds,
    xorq_circles_nearest_neighbors_preds,
    xorq_circles_linear_svm_preds,
    xorq_circles_rbf_svm_preds,
    xorq_circles_decision_tree_preds,
    xorq_circles_random_forest_preds,
    xorq_circles_neural_net_preds,
    xorq_circles_adaboost_preds,
    xorq_circles_naive_bayes_preds,
    xorq_linearly_separable_nearest_neighbors_preds,
    xorq_linearly_separable_linear_svm_preds,
    xorq_linearly_separable_rbf_svm_preds,
    xorq_linearly_separable_decision_tree_preds,
    xorq_linearly_separable_random_forest_preds,
    xorq_linearly_separable_neural_net_preds,
    xorq_linearly_separable_adaboost_preds,
    xorq_linearly_separable_naive_bayes_preds,
) = (
    comparators[ds_name].deferred_xorq_results[clf_name]["preds"]
    for ds_name in DATASET_NAMES
    for clf_name in CLASSIFIER_NAMES
)


# =========================================================================
# Run
# =========================================================================


def main():
    for ds_name in DATASET_NAMES:
        print(f"\n=== Dataset: {ds_name} ===")
        compare_results(comparators[ds_name])

    save_fig("imgs/plot_classifier_comparison.png", plot_results(comparators))


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
