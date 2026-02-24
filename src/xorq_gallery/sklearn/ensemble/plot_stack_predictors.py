"""Stacking Regressor Predictions
=================================

sklearn: Generate synthetic data (sinusoid + linear trend + drop), fit three base
regressors (linear ridge, spline ridge, histogram gradient boosting), stack them
with StackingRegressor using RidgeCV as final estimator, evaluate predictions.

xorq: Same base estimators and stacking regressor wrapped in Pipeline.from_instance,
fit/predict deferred via xorq, predictions match sklearn exactly.

Both produce identical predictions.

Dataset: Synthetic sinusoid with trend and drop
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import (
    fig_to_image,
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
        # Ensure estimators is a list before fitting
        estimators_list = (
            list(self.estimators)
            if not isinstance(self.estimators, list)
            else self.estimators
        )
        self._stacking_regressor = StackingRegressor(
            estimators=estimators_list, final_estimator=self.final_estimator
        )
        self._stacking_regressor.fit(X, y)
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
ROW_IDX = "row_idx"


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_data():
    """Generate synthetic data: sinusoid + linear trend + drop + noise."""
    rng = np.random.RandomState(RANDOM_STATE)
    X = rng.uniform(-3, 3, size=500)
    trend = 2.4 * X
    seasonal = 3.1 * np.sin(3.2 * X)
    drop = 10.0 * (X > 2).astype(float)
    sigma = 0.75 + 0.75 * X**2
    y = trend + seasonal - drop + rng.normal(loc=0.0, scale=np.sqrt(sigma))

    df = pd.DataFrame({FEATURE_COLS[0]: X, TARGET_COL: y})
    df[ROW_IDX] = range(len(df))

    print(f"Number of samples: {len(df)}")

    return df


# ---------------------------------------------------------------------------
# Shared model definitions
# ---------------------------------------------------------------------------


def _build_models():
    """Build base estimators and stacking regressor.

    Returns:
        Dict of model_name -> sklearn estimator instance
    """
    # Base estimators
    linear_ridge = SklearnPipeline(
        [
            ("standardscaler", StandardScaler()),
            ("ridgecv", RidgeCV()),
        ]
    )

    spline_ridge = SklearnPipeline(
        [
            ("splinetransformer", SplineTransformer(n_knots=6, degree=3)),
            ("polynomialfeatures", PolynomialFeatures(interaction_only=True)),
            ("ridgecv", RidgeCV()),
        ]
    )

    hgbt = HistGradientBoostingRegressor(random_state=0)

    # Create fresh instances for StackingRegressor
    stack_linear_ridge = SklearnPipeline(
        [
            ("standardscaler", StandardScaler()),
            ("ridgecv", RidgeCV()),
        ]
    )
    stack_spline_ridge = SklearnPipeline(
        [
            ("splinetransformer", SplineTransformer(n_knots=6, degree=3)),
            ("polynomialfeatures", PolynomialFeatures(interaction_only=True)),
            ("ridgecv", RidgeCV()),
        ]
    )
    stack_hgbt = HistGradientBoostingRegressor(random_state=0)

    estimators = [
        ("Linear Ridge", stack_linear_ridge),
        ("Spline Ridge", stack_spline_ridge),
        ("HGBT", stack_hgbt),
    ]

    stacking_regressor = StackingRegressor(
        estimators=estimators, final_estimator=RidgeCV()
    )

    return {
        "Linear Ridge": linear_ridge,
        "Spline Ridge": spline_ridge,
        "HGBT": hgbt,
        "Stacking": stacking_regressor,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_predictions(predictions_dict, X, y, title):
    """Build plot comparing predictions from base models and stacking regressor.

    Args:
        predictions_dict: Dict of model_name -> predictions array
        X: Input feature array
        y: True target values
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    model_names = ["Linear Ridge", "Spline Ridge", "HGBT", "Stacking"]

    for ax, name in zip(axes, model_names):
        y_pred = predictions_dict[name]

        ax.scatter(
            X[:, 0],
            y,
            s=6,
            alpha=0.35,
            linewidths=0,
            label="observed (sample)",
        )

        # Sort for line plotting
        sort_idx = np.argsort(X[:, 0])
        ax.plot(
            X[sort_idx, 0],
            y_pred[sort_idx],
            linewidth=2,
            alpha=0.9,
            label=name,
        )
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="lower right")

    fig.suptitle(title, y=1.0)
    fig.tight_layout()

    return fig


# =========================================================================
# SKLEARN WAY -- eager fit/predict on base models and stacking regressor
# =========================================================================


def sklearn_way(df, models):
    """Eager sklearn: fit base models and stacking regressor, predict.

    Returns:
        Dict of model_name -> predictions
    """
    X = df[[FEATURE_COLS[0]]].values
    y = df[TARGET_COL].values

    # Generate grid for smooth plotting
    x_plot = np.linspace(X.min() - 0.1, X.max() + 0.1, 500).reshape(-1, 1)

    # Fit all models on full dataset - smart flattening with comprehension
    model_names = ["Linear Ridge", "Spline Ridge", "HGBT", "Stacking"]
    fitted_models = {name: models[name].fit(X, y) for name in model_names}
    predictions = {name: fitted_models[name].predict(x_plot) for name in model_names}

    for name in model_names:
        print(f"  sklearn: {name:20s} fitted and predicted")

    return {"predictions": predictions, "X": x_plot, "y": y}


# =========================================================================
# XORQ WAY -- deferred fit/predict via xorq Pipeline
# =========================================================================


def xorq_way(df, models, x_plot):
    """Deferred xorq: wrap models in Pipeline.from_instance, fit/predict deferred.

    Returns deferred prediction expressions for each model.
    Nothing is executed until ``.execute()``.
    """
    con = xo.connect()
    data = con.register(df, "synthetic")

    # Register plot grid
    plot_df = pd.DataFrame({FEATURE_COLS[0]: x_plot[:, 0]})
    plot_data = con.register(plot_df, "plot_grid")

    # Linear Ridge
    linear_ridge_pipe = Pipeline.from_instance(models["Linear Ridge"])
    linear_ridge_fitted = linear_ridge_pipe.fit(
        data, features=FEATURE_COLS, target=TARGET_COL
    )
    linear_ridge_preds = linear_ridge_fitted.predict(plot_data, name=PRED_COL)

    # Spline Ridge
    spline_ridge_pipe = Pipeline.from_instance(models["Spline Ridge"])
    spline_ridge_fitted = spline_ridge_pipe.fit(
        data, features=FEATURE_COLS, target=TARGET_COL
    )
    spline_ridge_preds = spline_ridge_fitted.predict(plot_data, name=PRED_COL)

    # HGBT
    hgbt_pipe = Pipeline.from_instance(
        SklearnPipeline([("histgradientboostingregressor", models["HGBT"])])
    )
    hgbt_fitted = hgbt_pipe.fit(data, features=FEATURE_COLS, target=TARGET_COL)
    hgbt_preds = hgbt_fitted.predict(plot_data, name=PRED_COL)

    # Stacking - needs special wrapper
    wrapped_stacking = StackingRegressorWrapper(
        estimators=models["Stacking"].estimators,
        final_estimator=models["Stacking"].final_estimator,
    )
    stacking_pipeline = SklearnPipeline(
        [("stackingregressorwrapper", wrapped_stacking)]
    )
    stacking_pipe = Pipeline.from_instance(stacking_pipeline)
    stacking_fitted = stacking_pipe.fit(data, features=FEATURE_COLS, target=TARGET_COL)
    stacking_preds = stacking_fitted.predict(plot_data, name=PRED_COL)

    return {
        "Linear Ridge": linear_ridge_preds,
        "Spline Ridge": spline_ridge_preds,
        "HGBT": hgbt_preds,
        "Stacking": stacking_preds,
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    df = _load_data()

    # Generate grid for smooth plotting (in main, shared by both)
    X = df[[FEATURE_COLS[0]]].values
    x_plot = np.linspace(X.min() - 0.1, X.max() + 0.1, 500).reshape(-1, 1)
    y = df[TARGET_COL].values

    # Build separate model instances for sklearn and xorq to avoid state sharing
    sk_models = _build_models()
    xo_models = _build_models()

    print("\n=== SKLEARN WAY ===")
    sk_results = sklearn_way(df, sk_models)

    print("\n=== XORQ WAY ===")
    deferred = xorq_way(df, xo_models, x_plot)

    # Execute deferred predictions
    print("\n=== EXECUTING DEFERRED PREDICTIONS ===")
    xo_predictions = {}
    model_names = ["Linear Ridge", "Spline Ridge", "HGBT", "Stacking"]
    for name in model_names:
        pred_df = deferred[name].execute()
        xo_predictions[name] = pred_df[PRED_COL].values
        print(f"  xorq:   {name:20s} executed")

    # ---- Assert numerical equivalence BEFORE plotting ----
    print("\n=== ASSERTIONS ===")
    for name in model_names:
        sk_preds = sk_results["predictions"][name]
        xo_preds = xo_predictions[name]
        np.testing.assert_allclose(sk_preds, xo_preds, rtol=1e-5)
        print(
            f"  {name:20s}: predictions match (max_diff={np.max(np.abs(sk_preds - xo_preds)):.2e})"
        )

    print("Assertions passed: sklearn and xorq predictions match.")

    # Build sklearn plot
    sk_fig = _plot_predictions(
        sk_results["predictions"],
        sk_results["X"],
        sk_results["y"],
        "sklearn - Base Models vs Stacked Predictions",
    )

    # Build xorq plot
    xo_fig = _plot_predictions(
        xo_predictions,
        x_plot,
        y,
        "xorq - Base Models vs Stacked Predictions",
    )

    # Composite: sklearn (left) | xorq (right)
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].axis("off")
    axes[0].set_title("sklearn", fontsize=14, pad=10)

    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].axis("off")
    axes[1].set_title("xorq", fontsize=14, pad=10)

    fig.suptitle(
        "Stacking Regressor: sklearn vs xorq",
        fontsize=16,
        y=0.98,
    )
    fig.tight_layout()
    out = "imgs/plot_stack_predictors.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
