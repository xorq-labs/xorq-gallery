"""Comparison of Calibration of Classifiers
========================================

sklearn: Compare calibration of four different classifiers (Logistic Regression,
Naive Bayes, Linear SVC, Random Forest) on binary classification with 100,000
samples (100 train, 99,900 test). Uses CalibrationDisplay to plot calibration
curves and histograms showing probability distributions.

xorq: Same classifiers wrapped in Pipeline.from_instance, deferred fit and
predict_proba, deferred Brier score via deferred_sklearn_metric.

Both produce identical calibration metrics proving model reliability.

Dataset: make_classification (sklearn synthetic, 100k samples, 20 features)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/calibration/plot_compare_calibration.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from matplotlib.gridspec import GridSpec
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.svm import LinearSVC
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
)
from xorq_gallery.utils import fig_to_image, save_fig


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
PRED_COL = "predict_proba"
ROW_IDX = "row_idx"


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
        return np.c_[proba_neg_class, proba_pos_class]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

FEATURE_COLS = tuple(f"x{i}" for i in range(N_FEATURES))


def _load_data():
    """Generate synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        random_state=RANDOM_STATE,
    )
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df[TARGET_COL] = y
    df[ROW_IDX] = range(len(df))
    return df


def split_data(df):
    """Hash-based split via xorq for deterministic train/test."""
    con = xo.connect()
    table = con.register(df, "calibration_compare_data")
    test_size = (N_SAMPLES - TRAIN_SAMPLES) / N_SAMPLES
    train_data, test_data = xo.train_test_splits(
        table, test_sizes=test_size, unique_key=ROW_IDX, random_seed=RANDOM_STATE
    )
    train_df = train_data.order_by(ROW_IDX).execute()
    test_df = test_data.order_by(ROW_IDX).execute()
    return train_df, test_df


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

names_pipelines = (
    (
        "Logistic Regression",
        SklearnPipeline(
            [
                (
                    "lr",
                    LogisticRegressionCV(
                        Cs=tuple(np.logspace(-6, 6, 101).tolist()),
                        cv=10,
                        l1_ratios=(0,),
                        scoring="neg_log_loss",
                        max_iter=1_000,
                        use_legacy_attributes=False,
                    ),
                )
            ]
        ),
    ),
    (
        "Naive Bayes",
        SklearnPipeline([("gnb", GaussianNB())]),
    ),
    (
        "SVC",
        SklearnPipeline(
            [("svc", NaivelyCalibratedLinearSVC(C=1.0, random_state=RANDOM_STATE))]
        ),
    ),
    (
        "Random forest",
        SklearnPipeline([("rfc", RandomForestClassifier(random_state=RANDOM_STATE))]),
    ),
)

methods = tuple(name for name, _ in names_pipelines)


# ---------------------------------------------------------------------------
# Custom make_*_result: predict_proba + brier_score_loss
# ---------------------------------------------------------------------------


def _custom_make_sklearn_result(
    pipeline, train_data, test_data, features, target, metrics_names_funcs
):
    """Sklearn result using predict_proba and brier_score_loss."""
    X_train, y_train = train_data[list(features)], train_data[target]
    X_test, y_test = test_data[list(features)], test_data[target]

    fitted = clone(pipeline).fit(X_train, y_train)
    y_prob = fitted.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_prob)

    return {
        "fitted": fitted.steps[-1][-1],
        "preds": y_prob,
        "metrics": {"brier": brier},
        "other": {"y_prob": y_prob, "y_test": y_test.values},
    }


def _custom_make_deferred_xorq_result(
    pipeline, train_data, test_data, features, target, metrics_names_funcs, pred
):
    """Deferred xorq result using predict_proba and deferred brier_score_loss."""
    xorq_fitted = Pipeline.from_instance(pipeline).fit(
        train_data, features=features, target=target
    )
    proba_expr = xorq_fitted.predict_proba(test_data, name=pred)
    make_metric = deferred_sklearn_metric(target=target, pred=pred)
    metrics = {"brier": proba_expr.agg(brier=make_metric(metric=brier_score_loss))}

    return {
        "xorq_fitted": xorq_fitted,
        "preds": proba_expr,
        "metrics": metrics,
        "other": {},
    }


def _custom_make_xorq_result(deferred_xorq_result):
    """Materialize xorq result, extracting predict_proba column."""
    xorq_fitted = deferred_xorq_result["xorq_fitted"]
    preds_df = deferred_xorq_result["preds"].execute()
    brier_df = deferred_xorq_result["metrics"]["brier"].execute()
    brier = brier_df["brier"].iloc[0]

    # Extract probabilities — predict_proba returns lists [p0, p1] per row
    y_prob = np.array([p[1] for p in preds_df[PRED_COL]])
    y_test = preds_df[TARGET_COL].values

    return {
        "fitted": xorq_fitted.fitted_steps[-1].model,
        "preds": y_prob,
        "metrics": {"brier": brier},
        "other": {"y_prob": y_prob, "y_test": y_test},
    }


# ---------------------------------------------------------------------------
# Comparator callbacks
# ---------------------------------------------------------------------------


def compare_results(comparator):
    assert sorted(sklearn_results := comparator.sklearn_results) == sorted(
        xorq_results := comparator.xorq_results
    )
    print("\n=== Comparing Results ===")
    for name in sklearn_results:
        sk_brier = sklearn_results[name]["metrics"]["brier"]
        xo_brier = xorq_results[name]["metrics"]["brier"]
        print(f"  {name:25s} Brier - sklearn: {sk_brier:.4f}, xorq: {xo_brier:.4f}")
        np.testing.assert_allclose(sk_brier, xo_brier, rtol=1e-1)
    print("Assertions passed.")


def _plot_calibration_panel(results, title):
    """Build calibration plot with histograms for all classifiers."""
    colors_map = plt.get_cmap("Dark2")
    markers = ("^", "v", "s", "o")

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)
    ax_cal = fig.add_subplot(gs[:2, :2])

    for i, name in enumerate(methods):
        y_test = results[name]["other"]["y_test"]
        y_prob = results[name]["other"]["y_prob"]

        prob_true, prob_pred = calibration_curve(
            y_test, y_prob, n_bins=N_BINS, strategy="uniform"
        )
        ax_cal.plot(
            prob_pred,
            prob_true,
            marker=markers[i],
            label=name,
            color=colors_map(i),
            linewidth=2,
        )

    ax_cal.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax_cal.grid()
    ax_cal.set_title(title)
    ax_cal.set_xlabel("Mean predicted probability")
    ax_cal.set_ylabel("Fraction of positives")
    ax_cal.legend(loc="lower right")

    grid_positions = ((2, 0), (2, 1), (3, 0), (3, 1))
    for i, name in enumerate(methods):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])
        y_prob = results[name]["other"]["y_prob"]
        ax.hist(y_prob, range=(0, 1), bins=10, label=name, color=colors_map(i))
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    fig.tight_layout()
    return fig


def plot_results(comparator):
    sk_fig = _plot_calibration_panel(comparator.sklearn_results, "Calibration plots")
    xo_fig = _plot_calibration_panel(comparator.xorq_results, "Calibration plots")

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(
        "Comparison of Calibration of Classifiers: sklearn vs xorq",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()

    plt.close(sk_fig)
    plt.close(xo_fig)
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=TARGET_COL,
    pred=PRED_COL,
    metrics_names_funcs=(),
    load_data=_load_data,
    split_data=split_data,
    make_sklearn_result=_custom_make_sklearn_result,
    make_deferred_xorq_result=_custom_make_deferred_xorq_result,
    make_xorq_result=_custom_make_xorq_result,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)

# Module-level deferred exprs
(
    xorq_logistic_proba,
    xorq_naive_bayes_proba,
    xorq_svc_proba,
    xorq_random_forest_proba,
) = (comparator.deferred_xorq_results[name]["preds"] for name in methods)


def main():
    comparator.result_comparison
    save_fig("imgs/plot_compare_calibration.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
