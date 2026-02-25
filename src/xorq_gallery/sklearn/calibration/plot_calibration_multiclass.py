"""Probability Calibration for 3-class classification
=====================================================

sklearn: Generate 2000 synthetic 2D samples with 3 blob centers,
fit LogisticRegression uncalibrated and with CalibratedClassifierCV
(sigmoid), compare log_loss, plot arrows on a 2-simplex showing how
calibration shifts predicted probabilities.

xorq: Same classifiers wrapped in Pipeline.from_instance, deferred
predict_proba, deferred log_loss via deferred_sklearn_metric.
Deferred simplex plot via deferred_matplotlib_plot.

CalibratedClassifierCV has internal CV splitting that is sensitive to
row ordering -- marked [o] (order-sensitive).

Dataset: make_blobs (sklearn synthetic, 2000 samples, 2 features, 3 classes)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
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

RANDOM_STATE = 42
N_SAMPLES = 2000
N_FEATURES = 2
N_CLASSES = 3
CLUSTER_STD = 5.0
TEST_SIZE = 0.9
FEATURES = ("f0", "f1")
TARGET = "y"
ROW_IDX = "row_idx"
PROB_COLS = ("prob_0", "prob_1", "prob_2")


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Generate 3-class blob dataset matching sklearn example."""
    X, y = make_blobs(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        centers=N_CLASSES,
        random_state=RANDOM_STATE,
        cluster_std=CLUSTER_STD,
    )

    df = pd.DataFrame(X, columns=list(FEATURES))
    df[TARGET] = y
    df[ROW_IDX] = range(len(df))
    return df


def _build_classifiers():
    """Return (name, SklearnPipeline) pairs for uncalibrated and calibrated."""
    lr = LogisticRegression(C=1.0)
    cal_lr = CalibratedClassifierCV(LogisticRegression(C=1.0), cv=2, method="sigmoid")
    return [
        ("Uncalibrated", SklearnPipeline([("lr", lr)])),
        ("Sigmoid Calibrated", SklearnPipeline([("cal_lr", cal_lr)])),
    ]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

# Simplex corners for 3 classes
_CORNERS = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
CLASS_COLORS = ["red", "green", "blue"]


def _proba_to_simplex(proba):
    """Map 3-class probabilities to 2D simplex coordinates."""
    return proba @ _CORNERS


def _plot_simplex(ax, proba_uncal, proba_cal, y_true, title):
    """Plot arrows on 2-simplex from uncalibrated to calibrated probabilities."""
    pts_uncal = _proba_to_simplex(proba_uncal)
    pts_cal = _proba_to_simplex(proba_cal)

    # Draw simplex triangle
    triangle = plt.Polygon(_CORNERS, fill=False, edgecolor="black", linewidth=1.5)
    ax.add_patch(triangle)

    # Draw arrows coloured by true class
    for i in range(len(y_true)):
        ax.annotate(
            "",
            xy=pts_cal[i],
            xytext=pts_uncal[i],
            arrowprops=dict(
                arrowstyle="->",
                color=CLASS_COLORS[y_true[i]],
                alpha=0.4,
                linewidth=0.5,
            ),
        )

    # Label corners
    for k in range(N_CLASSES):
        ax.text(
            _CORNERS[k, 0],
            _CORNERS[k, 1],
            f" Class {k}",
            fontsize=10,
            fontweight="bold",
            color=CLASS_COLORS[k],
        )

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")


def _build_comparison_fig(
    proba_uncal, proba_cal, y_true, log_loss_uncal, log_loss_cal, title_prefix=""
):
    """Build a figure with simplex plot and log_loss annotation."""
    fig, ax = plt.subplots(figsize=(7, 6))
    _plot_simplex(
        ax,
        proba_uncal,
        proba_cal,
        y_true,
        f"{title_prefix}Sigmoid calibration on 2-simplex",
    )
    ax.text(
        0.02,
        0.02,
        f"Uncalibrated log_loss: {log_loss_uncal:.4f}\n"
        f"Calibrated log_loss:   {log_loss_cal:.4f}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    fig.tight_layout()
    return fig


@curry
def _build_xorq_simplex_plot(
    df_dummy,
    proba_uncal_frozen,
    proba_cal_frozen,
    y_true_frozen,
    log_loss_uncal,
    log_loss_cal,
):
    """Curried plot for deferred_matplotlib_plot.

    Arrays are passed as freeze()-wrapped lists for xorq hashability
    and converted back to numpy inside.
    """
    proba_uncal = np.array(proba_uncal_frozen)
    proba_cal = np.array(proba_cal_frozen)
    y_true = np.array(y_true_frozen)
    return _build_comparison_fig(
        proba_uncal,
        proba_cal,
        y_true,
        log_loss_uncal,
        log_loss_cal,
        title_prefix="xorq: ",
    )


# =========================================================================
# SKLEARN WAY -- eager fit, predict_proba, log_loss
# =========================================================================


def sklearn_way(df):
    """Eager sklearn: fit uncalibrated and calibrated LogReg, compute log_loss.

    Returns dict with probabilities and metrics for both classifiers.
    """
    X = df[list(FEATURES)].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    clfs = _build_classifiers()
    results = {}
    for name, pipe in clfs:
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)
        ll = log_loss(y_test, proba)
        results[name] = {"proba": proba, "log_loss": ll}
        print(f"  {name:25s} | log_loss = {ll:.4f}")

    results["y_test"] = y_test
    results["X_test"] = X_test
    return results


# =========================================================================
# XORQ WAY -- deferred fit/predict_proba, deferred metrics
# =========================================================================


def xorq_way(train_table, test_table):
    """Deferred xorq: Pipeline.from_instance + fit + predict_proba.

    Returns dict with deferred predict_proba expressions and deferred metrics.
    No .execute() calls here.
    """
    results = {}
    clfs = _build_classifiers()

    for name, pipe in clfs:
        xorq_pipe = Pipeline.from_instance(pipe)
        fitted = xorq_pipe.fit(train_table, features=FEATURES, target=TARGET)
        proba_expr = fitted.predict_proba(test_table)
        results[name] = {
            "proba_expr": proba_expr,
            "fitted": fitted,
        }
        print(f"  {name:25s} | deferred predict_proba created")

    return results


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()
    print(f"Generated {len(df)} samples with {N_CLASSES} classes")

    # Split into train/test DataFrames (matching sklearn's split)
    X = df[list(FEATURES)].values
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_df = pd.DataFrame(X_train, columns=list(FEATURES))
    train_df[TARGET] = y_train
    train_df[ROW_IDX] = range(len(train_df))

    test_df = pd.DataFrame(X_test, columns=list(FEATURES))
    test_df[TARGET] = y_test
    test_df[ROW_IDX] = range(len(test_df))

    con = xo.connect()
    train_table = con.register(train_df, "cal_train")
    test_table = con.register(test_df, "cal_test")

    print("\n=== SKLEARN WAY ===")
    sk_results = sklearn_way(df)

    print("\n=== XORQ WAY ===")
    xo_results = xorq_way(train_table, test_table)

    # --- Execute deferred predictions and assert equivalence ---
    print("\n=== ASSERTIONS ===")
    xo_probas = {}
    for name in ("Uncalibrated", "Sigmoid Calibrated"):
        xo_proba_df = xo_results[name]["proba_expr"].execute()
        # predict_proba returns a "predict_proba" column containing lists
        # of per-class probabilities -- stack into a 2D array.
        xo_proba = np.vstack(xo_proba_df["predict_proba"].values)
        xo_probas[name] = xo_proba

        xo_ll = log_loss(y_test, xo_proba)
        sk_ll = sk_results[name]["log_loss"]
        print(f"  {name:25s} | sklearn log_loss={sk_ll:.4f}, xorq log_loss={xo_ll:.4f}")
        np.testing.assert_allclose(
            sk_ll, xo_ll, rtol=0.05, err_msg=f"log_loss mismatch for {name}"
        )

    print("Assertions passed.")

    # --- Plotting ---
    print("\n=== PLOTTING ===")

    proba_uncal_sk = sk_results["Uncalibrated"]["proba"]
    proba_cal_sk = sk_results["Sigmoid Calibrated"]["proba"]
    ll_uncal_sk = sk_results["Uncalibrated"]["log_loss"]
    ll_cal_sk = sk_results["Sigmoid Calibrated"]["log_loss"]
    y_test_sk = sk_results["y_test"]

    # sklearn plot
    sk_fig = _build_comparison_fig(
        proba_uncal_sk,
        proba_cal_sk,
        y_test_sk,
        ll_uncal_sk,
        ll_cal_sk,
        title_prefix="sklearn: ",
    )

    # xorq deferred plot
    proba_uncal_xo = xo_probas["Uncalibrated"]
    proba_cal_xo = xo_probas["Sigmoid Calibrated"]
    ll_uncal_xo = log_loss(y_test, proba_uncal_xo)
    ll_cal_xo = log_loss(y_test, proba_cal_xo)

    dummy_table = con.register(pd.DataFrame({"dummy": [1]}), "dummy_cal_mc")
    xo_png = deferred_matplotlib_plot(
        dummy_table,
        _build_xorq_simplex_plot(
            proba_uncal_frozen=freeze(proba_uncal_xo.tolist()),
            proba_cal_frozen=freeze(proba_cal_xo.tolist()),
            y_true_frozen=freeze(y_test.tolist()),
            log_loss_uncal=ll_uncal_xo,
            log_loss_cal=ll_cal_xo,
        ),
        name="simplex_plot",
    ).execute()
    xo_img = load_plot_bytes(xo_png)

    # Composite: sklearn (left) | xorq (right)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(xo_img)
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    plt.close(sk_fig)

    fig.suptitle(
        "Probability Calibration (3-class): sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    out = "imgs/plot_calibration_multiclass.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComposite plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
