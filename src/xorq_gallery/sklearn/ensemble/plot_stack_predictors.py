"""Stacking Regressor Predictions
=================================

sklearn: Generate synthetic data (sinusoid + linear trend + drop), fit three base
regressors (linear ridge, spline ridge, histogram gradient boosting), stack them
with StackingRegressor using RidgeCV as final estimator, evaluate predictions.

xorq: Same base estimators and stacking regressor wrapped in Pipeline.from_instance
via StackingRegressorWrapper, fit/predict deferred, predictions match sklearn.

Both produce identical predictions.

Dataset: Synthetic sinusoid with trend and drop
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    make_sklearn_result as _make_sklearn_result,
    split_data_nop,
)
from xorq_gallery.utils import (
    fig_to_image,
    save_fig,
)


# ---------------------------------------------------------------------------
# Wrapper for StackingRegressor to handle tuple->list conversion
# ---------------------------------------------------------------------------


class StackingRegressorWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for StackingRegressor that ensures estimators param stays as list.

    xorq's attrs-based parameter storage converts lists to tuples for immutability,
    but sklearn's StackingRegressor validates that estimators must be a list.
    This wrapper fixes the type before calling fit/predict.
    """

    def __init__(self, estimators, final_estimator=None):
        self.estimators = estimators
        self.final_estimator = final_estimator

    def fit(self, X, y):
        estimators_list = (
            list(self.estimators)
            if not isinstance(self.estimators, list)
            else self.estimators
        )
        self._stacking_regressor = StackingRegressor(
            estimators=estimators_list, final_estimator=self.final_estimator
        )
        self._stacking_regressor.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self._stacking_regressor.predict(X)

    def get_params(self, deep=True):
        if hasattr(self, "_stacking_regressor"):
            return self._stacking_regressor.get_params(deep=deep)
        return {"estimators": self.estimators, "final_estimator": self.final_estimator}

    def set_params(self, **params):
        if "estimators" in params:
            self.estimators = params["estimators"]
        if "final_estimator" in params:
            self.final_estimator = params["final_estimator"]
        if hasattr(self, "_stacking_regressor"):
            self._stacking_regressor.set_params(**params)
        return self


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TARGET_COL = "y"
FEATURE_COLS = ("X",)
PRED_COL = "pred"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Generate synthetic data: sinusoid + linear trend + drop + noise."""
    rng = np.random.RandomState(RANDOM_STATE)
    X = rng.uniform(-3, 3, size=500)
    trend = 2.4 * X
    seasonal = 3.1 * np.sin(3.2 * X)
    drop = 10.0 * (X > 2).astype(float)
    sigma = 0.75 + 0.75 * X**2
    y = trend + seasonal - drop + rng.normal(loc=0.0, scale=np.sqrt(sigma))
    return pd.DataFrame({FEATURE_COLS[0]: X, TARGET_COL: y})


# ---------------------------------------------------------------------------
# Pipeline builders (sub-pipelines reused inside StackingRegressorWrapper)
# ---------------------------------------------------------------------------


def _linear_ridge():
    return SklearnPipeline(
        [("standardscaler", StandardScaler()), ("ridgecv", RidgeCV())]
    )


def _spline_ridge():
    return SklearnPipeline(
        [
            ("splinetransformer", SplineTransformer(n_knots=6, degree=3)),
            ("polynomialfeatures", PolynomialFeatures(interaction_only=True)),
            ("ridgecv", RidgeCV()),
        ]
    )


def _hgbt():
    return SklearnPipeline(
        [("histgradientboostingregressor", HistGradientBoostingRegressor(random_state=0))]
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_predictions(predictions_dict, X, y, title):
    """Build plot comparing predictions from base models and stacking regressor."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    model_names = [LINEAR_RIDGE, SPLINE_RIDGE, HGBT, STACKING]

    for ax, name in zip(axes, model_names):
        y_pred = predictions_dict[name]
        ax.scatter(X[:, 0], y, s=6, alpha=0.35, linewidths=0, label="observed (sample)")
        sort_idx = np.argsort(X[:, 0])
        ax.plot(X[sort_idx, 0], y_pred[sort_idx], linewidth=2, alpha=0.9, label=name)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="lower right")

    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# make_other override: store full fitted pipeline for smooth-grid prediction
# ---------------------------------------------------------------------------


def make_sklearn_other(fitted):
    # full pipeline needed in plot_results to predict on x_plot with preprocessing
    return {"full_pipeline": fitted}


make_sklearn_result = _make_sklearn_result(make_other=make_sklearn_other)


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
        sk_r2 = sklearn_result["metrics"]["r2"]
        xo_r2 = xorq_result["metrics"]["r2"]
        print(f"  {name:15s} r2 - sklearn: {sk_r2:.3f}, xorq: {xo_r2:.3f}")


def plot_results(comparator):
    X = comparator.df[[FEATURE_COLS[0]]].values
    x_plot = np.linspace(X.min() - 0.1, X.max() + 0.1, 500).reshape(-1, 1)
    plot_df = pd.DataFrame({FEATURE_COLS[0]: x_plot[:, 0]})
    y = comparator.df[TARGET_COL].values

    sk_predictions = {
        name: result["other"]["full_pipeline"].predict(x_plot)
        for name, result in comparator.sklearn_results.items()
    }
    xo_predictions = {
        name: comparator.deferred_xorq_results[name]["xorq_fitted"]
        .predict(xo.memtable(plot_df), name=PRED_COL)
        .execute()[PRED_COL]
        .values
        for name in methods
    }

    sk_fig = _plot_predictions(
        sk_predictions, x_plot, y, "sklearn - Base Models vs Stacked Predictions"
    )
    xo_fig = _plot_predictions(
        xo_predictions, x_plot, y, "xorq - Base Models vs Stacked Predictions"
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].axis("off")
    axes[0].set_title("sklearn", fontsize=14, pad=10)
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].axis("off")
    axes[1].set_title("xorq", fontsize=14, pad=10)
    fig.suptitle("Stacking Regressor: sklearn vs xorq", fontsize=16, y=0.98)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = (LINEAR_RIDGE, SPLINE_RIDGE, HGBT, STACKING) = (
    "Linear Ridge",
    "Spline Ridge",
    "HGBT",
    "Stacking",
)
names_pipelines = (
    (LINEAR_RIDGE, _linear_ridge()),
    (SPLINE_RIDGE, _spline_ridge()),
    (HGBT, _hgbt()),
    (
        STACKING,
        SklearnPipeline(
            [
                (
                    "stackingregressorwrapper",
                    StackingRegressorWrapper(
                        estimators=[
                            ("Linear Ridge", _linear_ridge()),
                            ("Spline Ridge", _spline_ridge()),
                            ("HGBT", HistGradientBoostingRegressor(random_state=0)),
                        ],
                        final_estimator=RidgeCV(),
                    ),
                )
            ]
        ),
    ),
)
metrics_names_funcs = (("r2", r2_score),)

comparator = SklearnXorqComparator(
    names_pipelines=names_pipelines,
    features=FEATURE_COLS,
    target=TARGET_COL,
    pred=PRED_COL,
    metrics_names_funcs=metrics_names_funcs,
    load_data=load_data,
    split_data=split_data_nop,
    make_sklearn_result=make_sklearn_result,
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)
# expose the exprs to invoke `xorq build plot_stack_predictors.py --expr $expr_name`
(xorq_linear_ridge_preds, xorq_spline_ridge_preds, xorq_hgbt_preds, xorq_stacking_preds) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    comparator.result_comparison
    save_fig("imgs/plot_stack_predictors.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
