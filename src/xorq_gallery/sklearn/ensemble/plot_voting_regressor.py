"""Voting Regressor Predictions
================================

sklearn: Load diabetes dataset, train three individual regressors
(GradientBoostingRegressor, RandomForestRegressor, LinearRegression),
train a VotingRegressor ensemble that averages their predictions,
and compare all predictions on the first 20 samples.

xorq: Same regressors and voting ensemble wrapped in Pipeline.from_instance
via VotingRegressorWrapper, fit/predict deferred, predictions match sklearn.

Both produce identical predictions.

Dataset: Diabetes (sklearn)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline as SklearnPipeline

from xorq_gallery.sklearn.sklearn_lib import (
    SklearnXorqComparator,
    split_data_nop,
)
from xorq_gallery.utils import (
    fig_to_image,
    save_fig,
)


# ---------------------------------------------------------------------------
# Wrapper for VotingRegressor to handle tuple->list conversion
# ---------------------------------------------------------------------------


class VotingRegressorWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for VotingRegressor that ensures estimators param stays as list.

    xorq's attrs-based parameter storage converts lists to tuples for immutability,
    but sklearn's VotingRegressor validates that estimators must be a list.
    This wrapper fixes the type before calling fit/predict.
    """

    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        estimators_list = (
            list(self.estimators)
            if not isinstance(self.estimators, list)
            else self.estimators
        )
        self._voting_regressor = VotingRegressor(estimators=estimators_list)
        self._voting_regressor.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self._voting_regressor.predict(X)

    def get_params(self, deep=True):
        if hasattr(self, "_voting_regressor"):
            return self._voting_regressor.get_params(deep=deep)
        return {"estimators": self.estimators}

    def set_params(self, **params):
        if "estimators" in params:
            self.estimators = params["estimators"]
        if hasattr(self, "_voting_regressor"):
            self._voting_regressor.set_params(**params)
        return self


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 1
N_DISPLAY_SAMPLES = 20
TARGET_COL = "target"
PRED_COL = "pred"
FEATURE_COLS = ("age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Load diabetes dataset as DataFrame."""
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return X.assign(**{TARGET_COL: y})


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_predictions(predictions_dict, y_true, title):
    """Build line plot comparing predictions from multiple models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = {
        "GradientBoosting": {
            "marker": "o",
            "color": "g",
            "label": "GradientBoostingRegressor",
        },
        "RandomForest": {"marker": "^", "color": "b", "label": "RandomForestRegressor"},
        "LinearRegression": {"marker": "s", "color": "y", "label": "LinearRegression"},
        "VotingRegressor": {
            "marker": "*",
            "color": "r",
            "markersize": 10,
            "label": "VotingRegressor",
        },
    }

    x_vals = np.arange(len(y_true))
    for model_name, preds in predictions_dict.items():
        style = styles.get(
            model_name, {"marker": "x", "color": "k", "label": model_name}
        )
        ax.plot(x_vals, preds, linestyle="", **style)

    ax.plot(x_vals, y_true, "k--", alpha=0.3, label="True values")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Predicted target")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

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
        sk_r2 = sklearn_result["metrics"]["r2"]
        xo_r2 = xorq_result["metrics"]["r2"]
        print(f"  {name:20s} r2 - sklearn: {sk_r2:.3f}, xorq: {xo_r2:.3f}")


def plot_results(comparator):
    y_true = comparator.df[TARGET_COL].iloc[:N_DISPLAY_SAMPLES].values

    sk_predictions = {
        name: result["preds"][:N_DISPLAY_SAMPLES]
        for name, result in comparator.sklearn_results.items()
    }
    xo_predictions = {
        name: result["preds"][PRED_COL].values[:N_DISPLAY_SAMPLES]
        for name, result in comparator.xorq_results.items()
    }

    sk_fig = _plot_predictions(
        sk_predictions, y_true, "sklearn - Individual and Voting Regressor Predictions"
    )
    xo_fig = _plot_predictions(
        xo_predictions, y_true, "xorq - Individual and Voting Regressor Predictions"
    )

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].axis("off")
    axes[0].set_title("sklearn", fontsize=12, pad=10)
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].axis("off")
    axes[1].set_title("xorq", fontsize=12, pad=10)
    fig.suptitle("Voting Regressor: sklearn vs xorq", fontsize=16, y=0.98)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

methods = (GB, RF, LR, VR) = (
    "GradientBoosting",
    "RandomForest",
    "LinearRegression",
    "VotingRegressor",
)
names_pipelines = (
    (
        GB,
        SklearnPipeline(
            [
                (
                    "gradientboostingregressor",
                    GradientBoostingRegressor(random_state=RANDOM_STATE),
                )
            ]
        ),
    ),
    (
        RF,
        SklearnPipeline(
            [
                (
                    "randomforestregressor",
                    RandomForestRegressor(random_state=RANDOM_STATE),
                )
            ]
        ),
    ),
    (
        LR,
        SklearnPipeline([("linearregression", LinearRegression())]),
    ),
    (
        VR,
        SklearnPipeline(
            [
                (
                    "votingregressorwrapper",
                    VotingRegressorWrapper(
                        estimators=[
                            ("gb", GradientBoostingRegressor(random_state=RANDOM_STATE)),
                            ("rf", RandomForestRegressor(random_state=RANDOM_STATE)),
                            ("lr", LinearRegression()),
                        ]
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
    compare_results_fn=compare_results,
    plot_results_fn=plot_results,
)
# expose the exprs to invoke `xorq build plot_voting_regressor.py --expr $expr_name`
(xorq_gb_preds, xorq_rf_preds, xorq_lr_preds, xorq_vr_preds) = (
    comparator.deferred_xorq_results[name]["preds"] for name in methods
)


def main():
    comparator.result_comparison
    save_fig("imgs/plot_voting_regressor.png", comparator.plot_results())


if __name__ in ("__pytest_main__",):
    main()
    pytest_examples_passed = True
