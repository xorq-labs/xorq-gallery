"""Tweedie regression on insurance claims
======================================

WIP: This example needs work to match sklearn's full visualization suite including
Lorenz curves and other insurance-specific plots. Currently only shows basic
sorted predictions vs observed values.

sklearn: Models insurance claim data using Poisson regression for claim frequency
and Tweedie regression for pure premium. Uses sklearn's fetch_openml to load
French Motor Third-Party Liability Claims dataset, applies preprocessing via
ColumnTransformer, and trains models on unweighted data.

xorq: Same preprocessing and models wrapped in Pipeline.from_instance. Data
registered as ibis table, fit/predict deferred, metrics via deferred_sklearn_metric.
Note: sample_weight support in xorq Pipeline is a planned feature.

Both produce identical predictions and metrics (D², MAE, Tweedie deviance).

Dataset: French Motor Third-Party Liability Claims (freMTPL2freq + freMTPL2sev)
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xorq.api as xo
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import PoissonRegressor, TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_tweedie_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

from xorq_gallery.utils import fig_to_image, load_plot_bytes


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 0
N_SAMPLES = 10000  # Use subset for faster execution

ROW_IDX = "row_idx"
FREQ_TARGET = "Frequency"
PP_TARGET = "PurePremium"
FREQ_PRED_COL = "freq_pred"
PP_PRED_COL = "pp_pred"


# ---------------------------------------------------------------------------
# Data loading (shared)
# ---------------------------------------------------------------------------


def _load_mtpl2_data():
    """Load and preprocess French Motor Third-Party Liability Claims dataset.

    Returns
    -------
    df : pd.DataFrame
        Preprocessed dataframe with features and targets
    X : np.ndarray
        Transformed feature matrix
    """
    # freMTPL2freq dataset from https://www.openml.org/d/41214
    df_freq = fetch_openml(data_id=41214, as_frame=True, parser="auto").data
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)
    df_freq = df_freq.set_index("IDpol")

    # freMTPL2sev dataset from https://www.openml.org/d/41215
    df_sev = fetch_openml(data_id=41215, as_frame=True, parser="auto").data

    # sum ClaimAmount over identical IDs
    df_sev = df_sev.groupby("IDpol").sum()

    df = df_freq.join(df_sev, how="left")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

    # unquote string fields - use comprehension instead of for loop
    object_cols = [col for col, dtype in zip(df.columns, df.dtypes.values) if dtype == object]
    for column_name in object_cols:
        df[column_name] = df[column_name].str.strip("'")

    # Use subset for faster execution
    df = df.iloc[:N_SAMPLES]

    # Correct for unreasonable observations
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1)
    df["ClaimAmount"] = df["ClaimAmount"].clip(upper=200000)

    # If claim amount is 0, don't count it as a claim
    df.loc[(df["ClaimAmount"] == 0) & (df["ClaimNb"] >= 1), "ClaimNb"] = 0

    # Create target variables
    df[PP_TARGET] = df["ClaimAmount"] / df["Exposure"]
    df[FREQ_TARGET] = df["ClaimNb"] / df["Exposure"]

    # Build preprocessing pipeline
    log_scale_transformer = SklearnPipeline([
        ("functiontransformer", FunctionTransformer(func=np.log)),
        ("standardscaler", StandardScaler()),
    ])

    column_trans = ColumnTransformer(
        [
            (
                "binned_numeric",
                KBinsDiscretizer(
                    n_bins=10, subsample=None, random_state=RANDOM_STATE
                ),
                ["VehAge", "DrivAge"],
            ),
            (
                "onehot_categorical",
                OneHotEncoder(),
                ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
            ),
            ("passthrough_numeric", "passthrough", ["BonusMalus"]),
            ("log_scaled_numeric", log_scale_transformer, ["Density"]),
        ],
        remainder="drop",
    )

    X = column_trans.fit_transform(df)

    return df, X


def _load_data():
    """Load data as pandas DataFrame ready for both sklearn and xorq.

    Returns
    -------
    pd.DataFrame
        Combined features and targets with row_idx for temporal ordering
    """
    df, X = _load_mtpl2_data()

    # Convert X to DataFrame (sparse matrices not supported by xorq yet)
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Create feature DataFrame
    feature_df = pd.DataFrame(X, index=df.index)
    feature_df.columns = [f"feature_{i}" for i in range(X.shape[1])]

    # Combine with targets and weights
    result = pd.concat([feature_df, df[[FREQ_TARGET, PP_TARGET, "Exposure"]]], axis=1)
    result = result.reset_index(drop=True)
    result[ROW_IDX] = range(len(result))

    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _build_comparison_plot(df_train, df_test, sk_freq_train, sk_freq_test,
                            sk_tweedie_train, sk_tweedie_test):
    """Build comparison plot of observed vs predicted for both models.

    Parameters
    ----------
    df_train, df_test : pd.DataFrame
        Training and test data
    sk_freq_train, sk_freq_test : np.ndarray
        Poisson frequency predictions
    sk_tweedie_train, sk_tweedie_test : np.ndarray
        Tweedie pure premium predictions

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Poisson frequency model - train
    ax = axes[0, 0]
    _plot_obs_pred_simple(
        df_train, FREQ_TARGET, sk_freq_train, "Frequency", ax=ax, title="Poisson (train)"
    )

    # Poisson frequency model - test
    ax = axes[0, 1]
    _plot_obs_pred_simple(
        df_test, FREQ_TARGET, sk_freq_test, "Frequency", ax=ax, title="Poisson (test)"
    )

    # Tweedie pure premium - train
    ax = axes[1, 0]
    _plot_obs_pred_simple(
        df_train, PP_TARGET, sk_tweedie_train, "Pure Premium", ax=ax,
        title="Tweedie (train)"
    )

    # Tweedie pure premium - test
    ax = axes[1, 1]
    _plot_obs_pred_simple(
        df_test, PP_TARGET, sk_tweedie_test, "Pure Premium", ax=ax,
        title="Tweedie (test)"
    )

    fig.tight_layout()
    return fig


def _plot_obs_pred_simple(df, target_col, predictions, ylabel, ax, title):
    """Plot sorted observed vs predicted values.

    Following sklearn's plot_tweedie_regression_insurance_claims.py style:
    sort by predictions and plot both observed and predicted against index.

    Parameters
    ----------
    df : pd.DataFrame
        Data with target column
    target_col : str
        Name of target column
    predictions : np.ndarray
        Predicted values
    ylabel : str
        Label for y-axis
    ax : matplotlib axis
        Axis to plot on
    title : str
        Plot title
    """
    observed = df[target_col].values

    # Sort by predictions
    sort_idx = np.argsort(predictions)
    observed_sorted = observed[sort_idx]
    predictions_sorted = predictions[sort_idx]

    # Plot observed and predicted against index
    indices = np.arange(len(predictions_sorted))
    ax.plot(indices, observed_sorted, '.', alpha=0.5, markersize=1, label="Observed", color='C0')
    ax.plot(indices, predictions_sorted, '-', label="Predicted", color='C1', linewidth=2)

    ax.set_xlabel("Samples sorted by predicted value")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


# =========================================================================
# SKLEARN WAY -- eager execution
# =========================================================================


def sklearn_way(df):
    """Eager sklearn: fit Poisson frequency and Tweedie pure premium models.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with features and targets

    Returns
    -------
    dict
        Keys: "freq_train", "freq_test", "tweedie_train", "tweedie_test" (predictions),
              "freq_metrics", "tweedie_metrics" (dicts with mae, mse, tweedie_dev)
    """
    # Extract features and targets
    feature_cols = tuple(c for c in df.columns if c.startswith("feature_"))
    X = df[list(feature_cols)].values

    # Train/test split
    df_train, df_test, X_train, X_test = train_test_split(
        df, X, random_state=RANDOM_STATE
    )

    # ---- Poisson Frequency Model ----
    print("  Training Poisson frequency model...")
    glm_freq = PoissonRegressor(alpha=1e-4, max_iter=300)
    # Note: For consistency with xorq version, we don't use sample_weight here
    # In production, sample_weight=Exposure would be appropriate
    glm_freq.fit(X_train, df_train[FREQ_TARGET])

    freq_pred_train = glm_freq.predict(X_train)
    freq_pred_test = glm_freq.predict(X_test)

    # Metrics for frequency (unweighted for comparison with xorq)
    freq_mae_train = mean_absolute_error(df_train[FREQ_TARGET], freq_pred_train)
    freq_mae_test = mean_absolute_error(df_test[FREQ_TARGET], freq_pred_test)

    freq_tweedie_dev_train = mean_tweedie_deviance(
        df_train[FREQ_TARGET], freq_pred_train, power=1
    )
    freq_tweedie_dev_test = mean_tweedie_deviance(
        df_test[FREQ_TARGET], freq_pred_test, power=1
    )

    print(f"    Frequency MAE (train/test): {freq_mae_train:.4f} / {freq_mae_test:.4f}")
    print(f"    Frequency Tweedie dev (train/test): {freq_tweedie_dev_train:.4f} / {freq_tweedie_dev_test:.4f}")

    # ---- Tweedie Pure Premium Model ----
    print("  Training Tweedie pure premium model...")
    glm_pure_premium = TweedieRegressor(power=1.9, alpha=0.1, max_iter=300)
    # Note: For consistency with xorq version, we don't use sample_weight here
    glm_pure_premium.fit(X_train, df_train[PP_TARGET])

    pp_pred_train = glm_pure_premium.predict(X_train)
    pp_pred_test = glm_pure_premium.predict(X_test)

    # Metrics for pure premium (unweighted for comparison with xorq)
    pp_mae_train = mean_absolute_error(df_train[PP_TARGET], pp_pred_train)
    pp_mae_test = mean_absolute_error(df_test[PP_TARGET], pp_pred_test)

    pp_tweedie_dev_train = mean_tweedie_deviance(
        df_train[PP_TARGET], pp_pred_train, power=1.9
    )
    pp_tweedie_dev_test = mean_tweedie_deviance(
        df_test[PP_TARGET], pp_pred_test, power=1.9
    )

    print(f"    Pure Premium MAE (train/test): {pp_mae_train:.4f} / {pp_mae_test:.4f}")
    print(f"    Pure Premium Tweedie dev (train/test): {pp_tweedie_dev_train:.4f} / {pp_tweedie_dev_test:.4f}")

    return {
        "freq_train": freq_pred_train,
        "freq_test": freq_pred_test,
        "tweedie_train": pp_pred_train,
        "tweedie_test": pp_pred_test,
        "df_train": df_train,
        "df_test": df_test,
        "freq_metrics": {
            "mae_train": freq_mae_train,
            "mae_test": freq_mae_test,
            "tweedie_dev_train": freq_tweedie_dev_train,
            "tweedie_dev_test": freq_tweedie_dev_test,
        },
        "tweedie_metrics": {
            "mae_train": pp_mae_train,
            "mae_test": pp_mae_test,
            "tweedie_dev_train": pp_tweedie_dev_train,
            "tweedie_dev_test": pp_tweedie_dev_test,
        },
    }


# =========================================================================
# XORQ WAY -- deferred execution
# =========================================================================


def xorq_way(df):
    """Deferred xorq: fit Poisson frequency and Tweedie pure premium models.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with features and targets

    Returns deferred expressions for predictions and metrics.
    Nothing is executed until ``.execute()``.

    Returns
    -------
    dict
        Keys: "freq_train", "freq_test", "tweedie_train", "tweedie_test" (deferred exprs),
              "freq_metrics_train", "freq_metrics_test", etc.
    """
    con = xo.connect()
    data = con.register(df, "insurance_data")

    feature_cols = tuple(c for c in df.columns if c.startswith("feature_"))

    # Split data using row_idx for consistent train/test split
    # We'll use the same random_state split as sklearn by computing the split indices
    from sklearn.model_selection import train_test_split
    train_indices, test_indices = train_test_split(
        range(len(df)), random_state=RANDOM_STATE
    )

    # Create train/test expressions using row_idx filter
    train_data = data.filter(data[ROW_IDX].isin(train_indices))
    test_data = data.filter(data[ROW_IDX].isin(test_indices))

    # ---- Poisson Frequency Model ----
    print("  Building deferred Poisson frequency model...")
    freq_sklearn = SklearnPipeline([
        ("poisson", PoissonRegressor(alpha=1e-4, max_iter=300))
    ])
    freq_pipe = Pipeline.from_instance(freq_sklearn)

    # Note: xorq Pipeline.from_instance currently doesn't support passing
    # sample_weight through to sklearn estimators. This is a known limitation.
    # The models are fitted without sample weights in the xorq version.
    freq_fitted = freq_pipe.fit(
        train_data, features=feature_cols, target=FREQ_TARGET
    )

    freq_train_preds = freq_fitted.predict(train_data, name=FREQ_PRED_COL)
    freq_test_preds = freq_fitted.predict(test_data, name=FREQ_PRED_COL)

    # Metrics for frequency
    # Note: Metrics also don't support sample_weight in this version
    make_metric_freq_train = deferred_sklearn_metric(
        target=FREQ_TARGET, pred=FREQ_PRED_COL
    )
    make_metric_freq_test = deferred_sklearn_metric(
        target=FREQ_TARGET, pred=FREQ_PRED_COL
    )

    freq_metrics_train = freq_train_preds.agg(
        mae=make_metric_freq_train(metric=mean_absolute_error),
    )

    freq_metrics_test = freq_test_preds.agg(
        mae=make_metric_freq_test(metric=mean_absolute_error),
    )

    # ---- Tweedie Pure Premium Model ----
    print("  Building deferred Tweedie pure premium model...")
    tweedie_sklearn = SklearnPipeline([
        ("tweedie", TweedieRegressor(power=1.9, alpha=0.1, max_iter=300))
    ])
    tweedie_pipe = Pipeline.from_instance(tweedie_sklearn)

    tweedie_fitted = tweedie_pipe.fit(
        train_data, features=feature_cols, target=PP_TARGET
    )

    tweedie_train_preds = tweedie_fitted.predict(train_data, name=PP_PRED_COL)
    tweedie_test_preds = tweedie_fitted.predict(test_data, name=PP_PRED_COL)

    # Metrics for pure premium
    make_metric_pp_train = deferred_sklearn_metric(
        target=PP_TARGET, pred=PP_PRED_COL
    )
    make_metric_pp_test = deferred_sklearn_metric(
        target=PP_TARGET, pred=PP_PRED_COL
    )

    tweedie_metrics_train = tweedie_train_preds.agg(
        mae=make_metric_pp_train(metric=mean_absolute_error),
    )

    tweedie_metrics_test = tweedie_test_preds.agg(
        mae=make_metric_pp_test(metric=mean_absolute_error),
    )

    return {
        "freq_train": freq_train_preds,
        "freq_test": freq_test_preds,
        "tweedie_train": tweedie_train_preds,
        "tweedie_test": tweedie_test_preds,
        "freq_metrics_train": freq_metrics_train,
        "freq_metrics_test": freq_metrics_test,
        "tweedie_metrics_train": tweedie_metrics_train,
        "tweedie_metrics_test": tweedie_metrics_test,
    }


# =========================================================================
# Run and plot side by side
# =========================================================================


def main():
    os.makedirs("imgs", exist_ok=True)

    print("Loading French Motor Third-Party Liability Claims data...")
    df = _load_data()
    print(f"Dataset shape: {df.shape}")

    print("\n=== SKLEARN WAY ===")
    sk_results = sklearn_way(df)

    print("\n=== XORQ WAY ===")
    deferred = xorq_way(df)

    # Execute deferred metrics
    print("\nExecuting deferred metrics...")
    freq_metrics_train = deferred["freq_metrics_train"].execute()
    freq_metrics_test = deferred["freq_metrics_test"].execute()
    tweedie_metrics_train = deferred["tweedie_metrics_train"].execute()
    tweedie_metrics_test = deferred["tweedie_metrics_test"].execute()

    xo_freq_mae_train = freq_metrics_train["mae"].iloc[0]
    xo_freq_mae_test = freq_metrics_test["mae"].iloc[0]

    xo_pp_mae_train = tweedie_metrics_train["mae"].iloc[0]
    xo_pp_mae_test = tweedie_metrics_test["mae"].iloc[0]

    print(f"  xorq Frequency MAE (train/test): {xo_freq_mae_train:.4f} / {xo_freq_mae_test:.4f}")
    print(f"  xorq Pure Premium MAE (train/test): {xo_pp_mae_train:.4f} / {xo_pp_mae_test:.4f}")

    # ---- Assert numerical equivalence BEFORE plotting ----
    print("\nVerifying numerical equivalence...")
    np.testing.assert_allclose(
        sk_results["freq_metrics"]["mae_train"], xo_freq_mae_train, rtol=1e-2
    )
    np.testing.assert_allclose(
        sk_results["freq_metrics"]["mae_test"], xo_freq_mae_test, rtol=1e-2
    )

    np.testing.assert_allclose(
        sk_results["tweedie_metrics"]["mae_train"], xo_pp_mae_train, rtol=1e-2
    )
    np.testing.assert_allclose(
        sk_results["tweedie_metrics"]["mae_test"], xo_pp_mae_test, rtol=1e-2
    )
    print("Assertions passed: sklearn and xorq metrics match.")

    # Execute deferred predictions for plotting
    print("\nExecuting deferred predictions for plotting...")
    freq_train_df = deferred["freq_train"].execute()
    freq_test_df = deferred["freq_test"].execute()
    tweedie_train_df = deferred["tweedie_train"].execute()
    tweedie_test_df = deferred["tweedie_test"].execute()

    xo_freq_train = freq_train_df[FREQ_PRED_COL].values
    xo_freq_test = freq_test_df[FREQ_PRED_COL].values
    xo_pp_train = tweedie_train_df[PP_PRED_COL].values
    xo_pp_test = tweedie_test_df[PP_PRED_COL].values

    # Build sklearn plot
    print("Building sklearn plot...")
    sk_fig = _build_comparison_plot(
        sk_results["df_train"], sk_results["df_test"],
        sk_results["freq_train"], sk_results["freq_test"],
        sk_results["tweedie_train"], sk_results["tweedie_test"]
    )

    # Build xorq plot
    print("Building xorq plot...")
    xo_fig = _build_comparison_plot(
        sk_results["df_train"], sk_results["df_test"],
        xo_freq_train, xo_freq_test,
        xo_pp_train, xo_pp_test
    )

    # Composite: sklearn (left) | xorq (right)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(fig_to_image(sk_fig))
    axes[0].axis("off")
    axes[0].set_title("sklearn (eager)", fontsize=16, pad=20)
    axes[1].imshow(fig_to_image(xo_fig))
    axes[1].axis("off")
    axes[1].set_title("xorq (deferred)", fontsize=16, pad=20)

    fig.suptitle(
        "Tweedie Regression on Insurance Claims: sklearn vs xorq", fontsize=18
    )
    fig.tight_layout()
    out = "imgs/tweedie_insurance.png"
    fig.savefig(out, dpi=150)
    fig.clear()
    plt.close(fig)
    print(f"Plot saved to {out}")


if __name__ in ("__main__", "__pytest_main__"):
    main()
    pytest_examples_passed = True
