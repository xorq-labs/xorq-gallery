"""Voting Regressor Predictions
================================

sklearn: Load diabetes dataset, train three individual regressors
(GradientBoostingRegressor, RandomForestRegressor, LinearRegression),
train a VotingRegressor ensemble that averages their predictions,
and compare all predictions on the first 20 samples.

xorq: Same regressors and voting ensemble wrapped in Pipeline.from_instance,
fit/predict deferred, predictions computed via deferred execution and match
sklearn exactly.

Both produce identical predictions.

Dataset: Diabetes (sklearn)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    deferred_matplotlib_plot,
    fig_to_image,
    load_plot_bytes,
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
        estimators_list = list(self.estimators) if not isinstance(self.estimators, list) else self.estimators
        self._voting_regressor = VotingRegressor(estimators=estimators_list)
        self._voting_regressor.fit(X, y)
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
ROW_IDX = "row_idx"


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Load diabetes dataset from sklearn."""
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    df = X.copy()
    df[TARGET_COL] = y
    df[ROW_IDX] = range(len(df))

    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")

    return df


# ---------------------------------------------------------------------------
# Shared model definitions
# ---------------------------------------------------------------------------


def _build_models():
    """Build individual regressors and voting ensemble."""
    reg1 = GradientBoostingRegressor(random_state=RANDOM_STATE)
    reg2 = RandomForestRegressor(random_state=RANDOM_STATE)
    reg3 = LinearRegression()

    ereg_reg1 = GradientBoostingRegressor(random_state=RANDOM_STATE)
    ereg_reg2 = RandomForestRegressor(random_state=RANDOM_STATE)
    ereg_reg3 = LinearRegression()

    # NOTE: VotingRegressor expects lists for estimators. xorq's attrs-based
    # storage converts lists to tuples, so VotingRegressorWrapper is needed.
    ereg = VotingRegressor(
        estimators=[("gb", ereg_reg1), ("rf", ereg_reg2), ("lr", ereg_reg3)]
    )

    return {
        "GradientBoosting": reg1,
        "RandomForest": reg2,
        "LinearRegression": reg3,
        "VotingRegressor": ereg,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_predictions(predictions_dict, y_true, title):
    """Build line plot comparing predictions from multiple models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = {
        "GradientBoosting": {"marker": "o", "color": "g", "label": "GradientBoostingRegressor"},
        "RandomForest": {"marker": "^", "color": "b", "label": "RandomForestRegressor"},
        "LinearRegression": {"marker": "s", "color": "y", "label": "LinearRegression"},
        "VotingRegressor": {"marker": "*", "color": "r", "markersize": 10, "label": "VotingRegressor"},
    }

    x_vals = np.arange(len(y_true))
    for model_name, preds in predictions_dict.items():
        style = styles.get(model_name, {"marker": "x", "color": "k", "label": model_name})
        ax.plot(x_vals, preds, linestyle="", **style)

    ax.plot(x_vals, y_true, "k--", alpha=0.3, label="True values")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Predicted target")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    return fig


# =========================================================================
# SKLEARN WAY -- eager fit/predict
# =========================================================================


def sklearn_way(df, models):
    """Eager sklearn: fit individual models and voting ensemble, predict."""
    feature_cols = tuple(col for col in df.columns if col not in (TARGET_COL, ROW_IDX))
    X = df[list(feature_cols)]
    y = df[TARGET_COL]

    predictions = {}
    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X.iloc[:N_DISPLAY_SAMPLES])
        predictions[name] = preds
        print(f"  sklearn: {name:20s} fitted and predicted")

    return predictions


# =========================================================================
# XORQ WAY -- deferred fit/predict via xorq Pipeline
# =========================================================================


def xorq_way(df, models):
    """Deferred xorq: wrap models in Pipeline.from_instance, fit/predict deferred."""
    con = xo.connect()
    data = con.register(df, "diabetes")
    feature_cols = tuple(col for col in df.columns if col not in (TARGET_COL, ROW_IDX))

    pred_exprs = {}
    for name, sklearn_model in models.items():
        # VotingRegressor needs wrapper for tuple->list conversion
        if isinstance(sklearn_model, VotingRegressor):
            wrapped = VotingRegressorWrapper(estimators=sklearn_model.estimators)
            step_name = "votingregressorwrapper"
        else:
            wrapped = sklearn_model
            step_name = type(sklearn_model).__name__.lower()

        sklearn_pipeline = SklearnPipeline([(step_name, wrapped)])
        xorq_pipe = Pipeline.from_instance(sklearn_pipeline)
        fitted = xorq_pipe.fit(data, features=feature_cols, target=TARGET_COL)
        preds = fitted.predict(data, name=PRED_COL)
        pred_exprs[name] = preds.order_by(ROW_IDX).limit(N_DISPLAY_SAMPLES)

    return pred_exprs


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()

    sk_models = _build_models()
    xo_models = _build_models()

    print("\n=== SKLEARN WAY ===")
    sk_predictions = sklearn_way(df, sk_models)

    print("\n=== XORQ WAY ===")
    deferred_preds = xorq_way(df, xo_models)

    # Execute deferred predictions
    print("\n=== EXECUTING DEFERRED PREDICTIONS ===")
    xo_predictions = {}
    for name, pred_expr in deferred_preds.items():
        pred_df = pred_expr.execute()
        xo_predictions[name] = pred_df[PRED_COL].values
        print(f"  xorq:   {name:20s} executed")

    # Assert
    print("\n=== ASSERTIONS ===")
    for name in sk_models.keys():
        sk_preds = sk_predictions[name]
        xo_preds = xo_predictions[name]
        np.testing.assert_allclose(sk_preds, xo_preds, rtol=1e-5)
        print(f"  {name:20s}: predictions match (max_diff={np.max(np.abs(sk_preds - xo_preds)):.2e})")

    print("Assertions passed: sklearn and xorq predictions match.")

    y_true = df[TARGET_COL].iloc[:N_DISPLAY_SAMPLES].values

    sk_fig = _plot_predictions(sk_predictions, y_true, "sklearn - Individual and Voting Regressor Predictions")
    xo_fig = _plot_predictions(xo_predictions, y_true, "xorq - Individual and Voting Regressor Predictions")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].axis("off")
    axes[0].set_title("sklearn", fontsize=12, pad=10)
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].axis("off")
    axes[1].set_title("xorq", fontsize=12, pad=10)

    fig.suptitle("Voting Regressor: sklearn vs xorq", fontsize=16, y=0.98)
    plt.tight_layout()
    out = "imgs/plot_voting_regressor.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
