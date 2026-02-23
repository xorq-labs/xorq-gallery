"""Comparing Nearest Neighbors with and without Neighborhood Components Analysis
=================================================================================

sklearn: Load iris dataset (2 features), train KNN and NCA+KNN pipelines with
StandardScaler, evaluate accuracy on stratified test split, plot decision
boundaries using DecisionBoundaryDisplay.

xorq: Same pipelines wrapped in Pipeline.from_instance, deferred fit/predict,
deferred accuracy metrics, deferred decision boundary plots via matplotlib.

Both produce identical accuracy scores.

Dataset: iris (sklearn)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline as SklearnPipeline
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

N_NEIGHBORS = 1
RANDOM_STATE = 42
TEST_SIZE = 0.7
H = 0.05  # meshgrid step size

# Color maps
CMAP_LIGHT = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
CMAP_BOLD = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Load iris dataset, select 2 features (sepal length, petal length).

    Returns pandas DataFrame with columns: x0, x1, y
    """
    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target

    # Only take two features for 2D visualization: sepal length [0], petal length [2]
    X = X[:, [0, 2]]

    # Create DataFrame
    df = pd.DataFrame(X, columns=["x0", "x1"])
    df["y"] = y

    return df


def _build_pipelines():
    """Build sklearn pipeline instances for KNN and NCA+KNN."""
    pipelines = {
        "KNN": SklearnPipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=N_NEIGHBORS)),
        ]),
        "NCA+KNN": SklearnPipeline([
            ("scaler", StandardScaler()),
            ("nca", NeighborhoodComponentsAnalysis(random_state=RANDOM_STATE)),
            ("knn", KNeighborsClassifier(n_neighbors=N_NEIGHBORS)),
        ]),
    }
    return pipelines


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_decision_boundary(ax, X, y, clf, title, score):
    """Plot decision boundary for a fitted sklearn classifier.

    Args:
        ax: matplotlib axis
        X: feature array (n_samples, 2)
        y: target array (n_samples,)
        clf: fitted sklearn classifier
        title: plot title
        score: accuracy score
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, H),
        np.arange(y_min, y_max, H)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.pcolormesh(xx, yy, Z, cmap=CMAP_LIGHT, alpha=0.8, shading="auto")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CMAP_BOLD, edgecolors="k", s=20)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)
    ax.text(
        0.9, 0.1, f"{score:.2f}",
        size=15, ha="center", va="center",
        transform=ax.transAxes
    )


# =========================================================================
# SKLEARN WAY -- eager fit/predict, decision boundary plots
# =========================================================================


def sklearn_way(df, pipelines):
    """Eager sklearn: fit KNN and NCA+KNN pipelines, compute accuracy,
    return fitted models and results for plotting.

    Args:
        df: pandas DataFrame with columns x0, x1, y
        pipelines: dict of pipeline_name -> sklearn Pipeline

    Returns:
        dict with keys: {KNN, NCA+KNN}
        Each result is dict: {clf, score, X_train, X_test, y_train, y_test, X_full, y_full}
    """
    X = df[["x0", "x1"]].values
    y = df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Fit KNN pipeline
    knn_clf = pipelines["KNN"]
    knn_clf.fit(X_train, y_train)
    knn_score = knn_clf.score(X_test, y_test)
    print(f"  sklearn: KNN             | acc = {knn_score:.4f}")

    # Fit NCA+KNN pipeline
    nca_knn_clf = pipelines["NCA+KNN"]
    nca_knn_clf.fit(X_train, y_train)
    nca_knn_score = nca_knn_clf.score(X_test, y_test)
    print(f"  sklearn: NCA+KNN         | acc = {nca_knn_score:.4f}")

    return {
        "KNN": {
            "clf": knn_clf,
            "score": knn_score,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_full": X,
            "y_full": y,
        },
        "NCA+KNN": {
            "clf": nca_knn_clf,
            "score": nca_knn_score,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_full": X,
            "y_full": y,
        },
    }


# =========================================================================
# XORQ WAY -- deferred fit/predict, deferred decision boundary plots
# =========================================================================


def xorq_way(df):
    """Deferred xorq: wrap KNN and NCA+KNN pipelines in Pipeline.from_instance,
    fit/predict deferred, compute deferred accuracy. Returns ONLY deferred expressions.

    Args:
        df: pandas DataFrame with columns x0, x1, y

    Returns:
        dict with keys: {KNN, NCA+KNN}
        Each result is dict: {preds: expr, metrics: expr, split_data: dict}
    """
    con = xo.connect()

    X = df[["x0", "x1"]].values
    y = df["y"].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Create train/test DataFrames
    train_df = pd.DataFrame(X_train, columns=["x0", "x1"])
    train_df["y"] = y_train
    test_df = pd.DataFrame(X_test, columns=["x0", "x1"])
    test_df["y"] = y_test

    # Register tables
    train_table = con.register(train_df, "train_iris")
    test_table = con.register(test_df, "test_iris")

    # Build fresh pipelines for xorq
    pipelines = _build_pipelines()

    # Fit KNN
    knn_pipe = Pipeline.from_instance(pipelines["KNN"])
    knn_fitted = knn_pipe.fit(train_table, features=("x0", "x1"), target="y")
    knn_preds = knn_fitted.predict(test_table, name="pred")
    knn_make_metric = deferred_sklearn_metric(target="y", pred="pred")
    knn_metrics = knn_preds.agg(acc=knn_make_metric(metric=accuracy_score))

    # Fit NCA+KNN
    nca_knn_pipe = Pipeline.from_instance(pipelines["NCA+KNN"])
    nca_knn_fitted = nca_knn_pipe.fit(train_table, features=("x0", "x1"), target="y")
    nca_knn_preds = nca_knn_fitted.predict(test_table, name="pred")
    nca_knn_make_metric = deferred_sklearn_metric(target="y", pred="pred")
    nca_knn_metrics = nca_knn_preds.agg(acc=nca_knn_make_metric(metric=accuracy_score))

    # Return deferred expressions + split data needed for plotting in main()
    return {
        "KNN": {
            "preds": knn_preds,
            "metrics": knn_metrics,
            "split_data": {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "X_full": X,
                "y_full": y,
                "sklearn_clf": pipelines["KNN"],
            },
        },
        "NCA+KNN": {
            "preds": nca_knn_preds,
            "metrics": nca_knn_metrics,
            "split_data": {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "X_full": X,
                "y_full": y,
                "sklearn_clf": pipelines["NCA+KNN"],
            },
        },
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()
    pipelines = _build_pipelines()

    print("=== SKLEARN WAY ===")
    sk_results = sklearn_way(df, pipelines)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(df)

    # Execute deferred metrics and assert equivalence
    print("\n=== ASSERTIONS ===")
    knn_xo_metrics_df = xo_results["KNN"]["metrics"].execute()
    knn_xo_score = knn_xo_metrics_df["acc"].iloc[0]
    print(f"  xorq:   KNN             | acc = {knn_xo_score:.4f}")
    np.testing.assert_allclose(sk_results["KNN"]["score"], knn_xo_score, rtol=1e-2)

    nca_knn_xo_metrics_df = xo_results["NCA+KNN"]["metrics"].execute()
    nca_knn_xo_score = nca_knn_xo_metrics_df["acc"].iloc[0]
    print(f"  xorq:   NCA+KNN         | acc = {nca_knn_xo_score:.4f}")
    np.testing.assert_allclose(sk_results["NCA+KNN"]["score"], nca_knn_xo_score, rtol=1e-2)

    print("Assertions passed: sklearn and xorq metrics match.")

    # Build deferred plots for xorq - HAPPENS IN MAIN, NOT IN xorq_way
    def _build_knn_plot(df):
        """Build decision boundary plot for KNN."""
        split_data = xo_results["KNN"]["split_data"]
        fitted_clf = split_data["sklearn_clf"].fit(split_data["X_train"], split_data["y_train"])
        score = fitted_clf.score(split_data["X_test"], split_data["y_test"])

        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_decision_boundary(
            ax, split_data["X_full"], split_data["y_full"], fitted_clf,
            f"KNN (k = {N_NEIGHBORS})",
            score
        )
        plt.tight_layout()
        return fig

    def _build_nca_knn_plot(df):
        """Build decision boundary plot for NCA+KNN."""
        split_data = xo_results["NCA+KNN"]["split_data"]
        fitted_clf = split_data["sklearn_clf"].fit(split_data["X_train"], split_data["y_train"])
        score = fitted_clf.score(split_data["X_test"], split_data["y_test"])

        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_decision_boundary(
            ax, split_data["X_full"], split_data["y_full"], fitted_clf,
            f"NCA+KNN (k = {N_NEIGHBORS})",
            score
        )
        plt.tight_layout()
        return fig

    # Execute deferred plots
    knn_png = deferred_matplotlib_plot(xo_results["KNN"]["preds"], _build_knn_plot).execute()
    nca_knn_png = deferred_matplotlib_plot(xo_results["NCA+KNN"]["preds"], _build_nca_knn_plot).execute()

    # Build sklearn plots
    fig_sk, axes_sk = plt.subplots(1, 2, figsize=(12, 5))

    knn_result = sk_results["KNN"]
    _plot_decision_boundary(
        axes_sk[0], knn_result["X_full"], knn_result["y_full"], knn_result["clf"],
        f"KNN (k = {N_NEIGHBORS})",
        knn_result["score"]
    )

    nca_knn_result = sk_results["NCA+KNN"]
    _plot_decision_boundary(
        axes_sk[1], nca_knn_result["X_full"], nca_knn_result["y_full"], nca_knn_result["clf"],
        f"NCA+KNN (k = {N_NEIGHBORS})",
        nca_knn_result["score"]
    )

    plt.suptitle("sklearn: KNN vs NCA+KNN", fontsize=14)
    plt.tight_layout()

    # Build xorq plots grid
    fig_xo, axes_xo = plt.subplots(1, 2, figsize=(12, 5))

    axes_xo[0].imshow(load_plot_bytes(knn_png))
    axes_xo[0].axis("off")

    axes_xo[1].imshow(load_plot_bytes(nca_knn_png))
    axes_xo[1].axis("off")

    plt.suptitle("xorq: KNN vs NCA+KNN", fontsize=14)
    plt.tight_layout()

    # Composite side-by-side
    sk_img = fig_to_image(fig_sk)
    xo_img = fig_to_image(fig_xo)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes[0].imshow(sk_img)
    axes[0].set_title("sklearn", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(xo_img)
    axes[1].set_title("xorq", fontsize=12)
    axes[1].axis("off")

    plt.suptitle(
        "Comparing Nearest Neighbors with/without NCA: sklearn vs xorq",
        fontsize=16
    )
    plt.tight_layout()
    out = "imgs/plot_nca_classification.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
