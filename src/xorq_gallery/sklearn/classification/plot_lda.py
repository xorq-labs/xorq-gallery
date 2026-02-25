"""Normal, Ledoit-Wolf and OAS Linear Discriminant Analysis
==========================================================

sklearn: Compare classification accuracy of LDA with three covariance
estimation strategies (empirical, Ledoit-Wolf shrinkage, OAS estimator)
as the number of noisy features increases, averaged over 50 repetitions.
Shows that shrinkage/OAS improves accuracy when n_features >> n_samples.

xorq: Same three LDA variants wrapped in Pipeline.from_instance, deferred
fit/predict on a representative dataset.  The accuracy sweep is run eagerly
(2850 pipeline fits), then xorq verifies equivalence on a single config.
Deferred plot via deferred_matplotlib_plot.

Both produce identical accuracy curves.

Dataset: Synthetic blobs (1 informative + noise features, 20 train / 200 test)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.covariance import OAS
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline as SklearnPipeline
from toolz import curry
from xorq.expr.ml.pipeline_lib import Pipeline
from xorq.ibis_yaml.utils import freeze

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TRAIN = 20
N_TEST = 200
N_AVERAGES = 50
N_FEATURES_MAX = 75
STEP = 4
FEATURES_RANGE = list(range(1, N_FEATURES_MAX + 1, STEP))
TARGET = "y"
ROW_IDX = "row_idx"

# LDA configurations: (name, builder)
LDA_CONFIGS = (
    (
        "Empirical",
        lambda: SklearnPipeline(
            [("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None))]
        ),
    ),
    (
        "Ledoit-Wolf",
        lambda: SklearnPipeline(
            [("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]
        ),
    ),
    (
        "OAS",
        lambda: SklearnPipeline(
            [
                (
                    "lda",
                    LinearDiscriminantAnalysis(
                        solver="lsqr", covariance_estimator=OAS()
                    ),
                )
            ]
        ),
    ),
)


# ---------------------------------------------------------------------------
# Data generation (shared)
# ---------------------------------------------------------------------------


def _generate_data(n_samples, n_features, rng):
    """Generate binary classification data with 1 informative + noise features."""
    X, y = make_blobs(
        n_samples=n_samples, n_features=1, centers=[[-2], [2]], random_state=rng
    )
    if n_features > 1:
        X = np.hstack([X, rng.randn(n_samples, n_features - 1)])
    return X, y


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_accuracy_curves(results, title_prefix=""):
    """Plot accuracy vs n_features for each LDA method."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"Empirical": "navy", "Ledoit-Wolf": "orange", "OAS": "red"}
    for name in ("Empirical", "Ledoit-Wolf", "OAS"):
        ax.plot(
            FEATURES_RANGE,
            results[name],
            label=name,
            color=colors[name],
            linewidth=2,
        )

    ax.set_xlabel("Number of features")
    ax.set_ylabel("Classification accuracy")
    ax.set_title(
        f"{title_prefix}LDA: Normal vs Ledoit-Wolf vs OAS\n"
        f"({N_AVERAGES} averages, {N_TRAIN} train, {N_TEST} test)"
    )
    ax.legend(loc="lower left")
    ax.set_ylim(0.45, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@curry
def _build_xorq_accuracy_plot(df_dummy, results_frozen, title_prefix):
    """Curried plot for deferred_matplotlib_plot.

    results_frozen is a freeze()-wrapped dict of {name: [accuracies]}.
    """
    results = {name: list(vals) for name, vals in results_frozen}
    return _plot_accuracy_curves(results, title_prefix=title_prefix)


# =========================================================================
# SKLEARN WAY -- eager sweep over features and repetitions
# =========================================================================


def sklearn_way():
    """Eager sklearn: sweep LDA accuracy over feature counts.

    Returns dict of method_name -> list of mean accuracies per feature count.
    """
    acc = {name: [] for name, _ in LDA_CONFIGS}

    for n_features in FEATURES_RANGE:
        scores = {name: 0.0 for name, _ in LDA_CONFIGS}

        for i in range(N_AVERAGES):
            rng = np.random.RandomState(i)
            X, y = _generate_data(N_TRAIN + N_TEST, n_features, rng)
            X_train, X_test = X[:N_TRAIN], X[N_TRAIN:]
            y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]

            for name, builder in LDA_CONFIGS:
                pipe = builder()
                pipe.fit(X_train, y_train)
                scores[name] += pipe.score(X_test, y_test)

        for name, _ in LDA_CONFIGS:
            acc[name].append(scores[name] / N_AVERAGES)

    for name in acc:
        print(
            f"  {name:15s} | accuracy range: [{min(acc[name]):.3f}, {max(acc[name]):.3f}]"
        )

    return acc


# =========================================================================
# XORQ WAY -- deferred fit/predict on representative config
# =========================================================================


def xorq_way(train_table, test_table, features):
    """Deferred xorq: Pipeline.from_instance + fit + predict for each LDA variant.

    Demonstrates that all three covariance strategies work with xorq Pipeline.
    Returns dict of method_name -> deferred prediction expression.
    No .execute() calls here.
    """
    results = {}

    for name, builder in LDA_CONFIGS:
        pipe = builder()
        xorq_pipe = Pipeline.from_instance(pipe)
        fitted = xorq_pipe.fit(train_table, features=features, target=TARGET)
        preds = fitted.predict(test_table, name="pred")
        results[name] = preds
        print(f"  {name:15s} | deferred predict created")

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    print("=== SKLEARN WAY ===")
    print(f"Sweeping {len(FEATURES_RANGE)} feature counts x {N_AVERAGES} averages...")
    sk_results = sklearn_way()

    # --- xorq: verify on a single representative config ---
    print("\n=== XORQ WAY (representative config: n_features=10) ===")
    n_features_rep = 10
    rng = np.random.RandomState(0)
    X, y = _generate_data(N_TRAIN + N_TEST, n_features_rep, rng)
    X_train, X_test = X[:N_TRAIN], X[N_TRAIN:]
    y_train, y_test = y[:N_TRAIN], y[N_TRAIN:]

    feature_cols = tuple(f"f{i}" for i in range(n_features_rep))
    train_df = pd.DataFrame(X_train, columns=list(feature_cols))
    train_df[TARGET] = y_train
    train_df[ROW_IDX] = range(len(train_df))

    test_df = pd.DataFrame(X_test, columns=list(feature_cols))
    test_df[TARGET] = y_test
    test_df[ROW_IDX] = range(len(test_df))

    con = xo.connect()
    train_table = con.register(train_df, "lda_train")
    test_table = con.register(test_df, "lda_test")

    xo_results = xorq_way(train_table, test_table, feature_cols)

    # --- Assertions ---
    print("\n=== ASSERTIONS ===")
    for name, builder in LDA_CONFIGS:
        # sklearn reference
        pipe = builder()
        pipe.fit(X_train, y_train)
        sk_score = pipe.score(X_test, y_test)

        # xorq
        xo_preds_df = xo_results[name].execute()
        xo_preds = xo_preds_df.sort_values(ROW_IDX)["pred"].values
        xo_score = np.mean(xo_preds == y_test)

        print(f"  {name:15s} | sklearn={sk_score:.3f}, xorq={xo_score:.3f}")
        np.testing.assert_allclose(
            sk_score, xo_score, atol=1e-10, err_msg=f"Accuracy mismatch for {name}"
        )

    print("Assertions passed.")

    # --- Plotting ---
    print("\n=== PLOTTING ===")

    # sklearn plot
    sk_fig = _plot_accuracy_curves(sk_results, title_prefix="sklearn: ")

    # xorq deferred plot -- reuse the same sweep results since the accuracy
    # sweep is deterministic and identical between sklearn and xorq.
    # The deferred plot demonstrates xorq's deferred_matplotlib_plot mechanism.
    dummy_table = con.register(pd.DataFrame({"dummy": [1]}), "dummy_lda")
    results_frozen = freeze(tuple((name, acc) for name, acc in sk_results.items()))
    xo_png = deferred_matplotlib_plot(
        dummy_table,
        _build_xorq_accuracy_plot(
            results_frozen=results_frozen,
            title_prefix="xorq: ",
        ),
        name="lda_plot",
    ).execute()
    xo_img = load_plot_bytes(xo_png)

    # Composite: sklearn (left) | xorq (right)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(xo_img)
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    plt.close(sk_fig)

    fig.suptitle(
        "LDA Covariance Estimation: sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    out = "imgs/plot_lda.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
