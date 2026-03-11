"""Recursive feature elimination
===============================

sklearn: Load digits dataset, apply RFE with LogisticRegression using MinMaxScaler
preprocessing to determine pixel importance rankings, visualize rankings as a heatmap
with numerical annotations.

xorq: Same RFE pipeline wrapped in Pipeline.from_instance, deferred fit to extract
feature rankings via xorq_fitted.fitted_steps[1].model.ranking_.

Both produce identical ranking values.

Dataset: load_digits (sklearn handwritten digits 0-9)

Source: https://github.com/scikit-learn/scikit-learn/blob/main/examples/feature_selection/plot_rfe_digits.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import MinMaxScaler
from toolz import curry
from xorq.common.utils.func_utils import return_constant
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    split_data_nop,
)
from xorq_gallery.utils import (
    fig_to_image,
    save_fig,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FEATURES = 64
IMAGE_SHAPE = (8, 8)
FEATURE_COLS = tuple(f"f{i}" for i in range(N_FEATURES))
TARGET_COL = "target"
PRED_COL = "pred"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Load the digits dataset and return as DataFrame."""
    digits = load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    return pd.DataFrame(X, columns=FEATURE_COLS).assign(**{TARGET_COL: digits.target})


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_ranking(ranking, title="Ranking of pixels with RFE\n(Logistic Regression)"):
    """Build ranking heatmap visualization."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.matshow(ranking, cmap=plt.cm.Blues)
    for i in range(ranking.shape[0]):
        for j in range(ranking.shape[1]):
            ax.text(
                j,
                i,
                str(int(ranking[i, j])),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


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
        print(f"  {name} accuracy - sklearn: {sk_acc:.4f}, xorq: {xo_acc:.4f}")

    sk_ranking = sklearn_results[RFE_NAME]["fitted"].ranking_.reshape(IMAGE_SHAPE)
    xo_fitted = comparator.deferred_xorq_results[RFE_NAME]["xorq_fitted"]
    xo_ranking = xo_fitted.fitted_steps[1].model.ranking_.reshape(IMAGE_SHAPE)
    np.testing.assert_array_equal(sk_ranking, xo_ranking)
    print("Rankings match.")


def plot_results(comparator):
    # sklearn ranking: result["fitted"] IS the RFE instance (last pipeline step)
    sk_ranking = comparator.sklearn_results[RFE_NAME]["fitted"].ranking_.reshape(
        IMAGE_SHAPE
    )

    # xorq ranking: access via fitted_steps on the xorq fitted pipeline
    xo_fitted = comparator.deferred_xorq_results[RFE_NAME]["xorq_fitted"]
    xo_ranking = xo_fitted.fitted_steps[1].model.ranking_.reshape(IMAGE_SHAPE)

    sk_fig = _plot_ranking(sk_ranking)
    xo_fig = _plot_ranking(xo_ranking)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].set_title("sklearn")
    axes[0].axis("off")
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].set_title("xorq")
    axes[1].axis("off")
    fig.suptitle("RFE Pixel Rankings on Digits: sklearn vs xorq", fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# xorq predict override: xorq's registry doesn't handle RFE, so we fit via
# Pipeline.from_instance and manually apply the fitted sklearn steps.
# ---------------------------------------------------------------------------


@curry
def _make_rfe_deferred_xorq_result(
    pipeline,
    train_data,
    test_data,
    features,
    target,
    metrics_names_funcs,
    pred,
    make_other=return_constant(None),
):
    xorq_fitted = Pipeline.from_instance(pipeline).fit(
        train_data, features=features, target=target
    )
    # xorq can't dispatch predict for RFE; apply fitted sklearn steps manually
    test_df = test_data.execute()
    X_test = test_df[list(features)].values
    y_test = test_df[target].values
    fitted_scaler = xorq_fitted.fitted_steps[0].model
    fitted_rfe = xorq_fitted.fitted_steps[1].model
    y_pred = fitted_rfe.predict(fitted_scaler.transform(X_test))
    preds_df = pd.DataFrame({target: y_test, pred: y_pred})
    metrics = {
        name: metric_fn(y_test, y_pred) for name, metric_fn in metrics_names_funcs
    }
    return {
        "xorq_fitted": xorq_fitted,
        "preds": preds_df,
        "metrics": metrics,
        "other": {},
    }


def _make_rfe_xorq_result(deferred_xorq_result):
    return {
        "fitted": deferred_xorq_result["xorq_fitted"].fitted_steps[-1].model,
        "preds": deferred_xorq_result["preds"],
        "metrics": deferred_xorq_result["metrics"],
    }


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = (RFE_NAME,) = ("RFE",)
names_pipelines = (
    (
        RFE_NAME,
        SklearnPipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "rfe",
                    RFE(estimator=LogisticRegression(), n_features_to_select=1, step=1),
                ),
            ]
        ),
    ),
)
metrics_names_funcs = (("accuracy", accuracy_score),)

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=TARGET_COL,
    pred=PRED_COL,
    metrics_names_funcs=metrics_names_funcs,
    load_data=load_data,
    split_data=split_data_nop,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
    make_deferred_xorq_result=_make_rfe_deferred_xorq_result,
    make_xorq_result=_make_rfe_xorq_result,
)
# Note: xorq build is not supported for RFE (xorq registry can't handle RFE predict)


def main():
    comparator.result_comparison
    save_fig("imgs/plot_rfe_digits.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
