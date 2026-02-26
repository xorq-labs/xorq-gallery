"""Shared library for sklearn gallery examples.

SklearnXorqComparator is a frozen attrs shell that holds the declaration
(pipelines, data, column config) and orchestrates the repeatable structure
of an sklearn-vs-xorq comparison. All behavior — splitting, expr building,
computing, plotting — is provided by injectable callables defined inline in
the example file, keeping the operations visible and transparent to the reader.
"""

from __future__ import annotations

import pandas as pd
import xorq.api as xo
from attr.validators import deep_iterable, instance_of, is_callable, optional
from attrs import field, frozen
from sklearn.base import clone
from xorq.expr.ml.pipeline_lib import Pipeline


@frozen
class SklearnXorqComparator:
    """Shell for an sklearn-vs-xorq gallery example.

    Holds the declaration (what pipelines, what data, what columns) and
    orchestrates the repeatable structure. All behavior — splitting, expr
    building, computing, plotting — is defined by callables in the example file.

    Parameters
    ----------
    name : str
        Example name. Also used as the ibis table name for registration.
    named_pipelines : tuple[tuple[str, SklearnPipeline], ...]
        Named sklearn pipelines to compare. Always use explicit named steps.
        Tuple-of-tuples for hashability on frozen class.
    df : pd.DataFrame
        The dataset. Passed in directly — ``_load_data()`` is called inline
        in the example so the reader sees the data source.
    features : tuple[str, ...]
        Feature column names.
    target : str or None
        Target column name. None for unsupervised examples.
    pred_col : str
        Base prediction column name (default ``"pred"``).
    metrics : tuple[tuple[str, Callable], ...]
        Named metric functions, e.g. ``(("r2", r2_score),)``.
        Tuple-of-tuples for hashability.
    sklearn_split_fn : callable or None
        ``(df, features, target) -> split_result``. Defined in the example.
        Return type is up to the example (could be train/test DFs, or
        X_train/X_test/y_train/y_test arrays).
    xorq_split_fn : callable or None
        ``(input_expr, features, target) -> (train_expr, test_expr)``.
        Defined in the example.
    build_exprs_fn : callable or None
        ``(sklearn_pipeline, train_expr, test_expr, features, target, pred_name) -> dict[str, Expr]``.
        Defined in the example. Receives the raw sklearn pipeline and
        does ``Pipeline.from_instance``, fit, predict, metrics visibly.
        Returns dict with keys like ``"fitted_pipeline"``, ``"preds"``,
        ``"metrics"``.
    compute_sklearn_fn : callable or None
        ``(comparator) -> results_dict``. Defined in the example.
        Contains the eager sklearn fit/predict/metric logic.
    compute_xorq_fn : callable or None
        ``(comparator) -> results_dict``. Defined in the example.
        Contains the deferred xorq execution logic.
    build_assertions_fn : callable or None
        ``(sk_results, xo_results, comparator) -> list[tuple[str, DataFrame, DataFrame]]``.
        Defined in the example. Returns ``(label, sk_df, xo_df)`` tuples
        with informative column names so failures pinpoint the source.
    plot_fn : callable or None
        ``(sk_results, xo_results, comparator) -> None``.
        Defined in the example. Handles figure creation, compositing, saving.
    """

    name = field(validator=instance_of(str))
    named_pipelines = field(
        validator=deep_iterable(instance_of(tuple), instance_of(tuple))
    )
    df = field(validator=instance_of(pd.DataFrame), eq=False, hash=False)
    features = field(validator=instance_of(tuple))
    target = field(validator=optional(instance_of(str)), default=None)
    pred_col = field(validator=instance_of(str), default="pred")
    metrics = field(
        validator=deep_iterable(instance_of(tuple), instance_of(tuple)),
        default=(),
    )

    # Injectable callables — defined inline in the example
    sklearn_split_fn = field(
        validator=optional(is_callable()), default=None, eq=False, hash=False
    )
    xorq_split_fn = field(
        validator=optional(is_callable()), default=None, eq=False, hash=False
    )
    build_exprs_fn = field(
        validator=optional(is_callable()), default=None, eq=False, hash=False
    )
    compute_sklearn_fn = field(
        validator=optional(is_callable()), default=None, eq=False, hash=False
    )
    compute_xorq_fn = field(
        validator=optional(is_callable()), default=None, eq=False, hash=False
    )
    build_assertions_fn = field(
        validator=optional(is_callable()), default=None, eq=False, hash=False
    )
    plot_fn = field(
        validator=optional(is_callable()), default=None, eq=False, hash=False
    )

    # ── Data access ────────────────────────────────────────────

    @property
    def pipelines(self):
        """Pipeline dict from the tuple-of-tuples."""
        return dict(self.named_pipelines)

    @property
    def input_expr(self):
        """Register df as an ibis table. Cheap — builds expr tree, no execution."""
        return xo.connect().register(self.df, self.name)

    @property
    def X(self):
        """Feature matrix as numpy array."""
        return self.df[list(self.features)].values

    @property
    def y(self):
        """Target array as numpy array. None for unsupervised."""
        return self.df[self.target].values if self.target else None

    @property
    def names(self):
        """Pipeline names as a tuple."""
        return tuple(name for name, _ in self.named_pipelines)

    # ── Pipeline access ────────────────────────────────────────

    def sklearn_pipeline(self, name):
        """Fresh clone of a named pipeline. Caller fits it explicitly."""
        return clone(self.pipelines[name])

    def xorq_pipeline(self, name):
        """Wrapped deferred pipeline. Caller fits it explicitly."""
        return Pipeline.from_instance(self.pipelines[name])

    # ── Splits ─────────────────────────────────────────────────

    @property
    def sklearn_split(self):
        """Delegate to example's sklearn_split_fn, or return (df, df)."""
        if self.sklearn_split_fn:
            return self.sklearn_split_fn(self.df, self.features, self.target)
        return self.df, self.df

    @property
    def xorq_split(self):
        """Delegate to example's xorq_split_fn, or return (input_expr, input_expr)."""
        if self.xorq_split_fn:
            return self.xorq_split_fn(self.input_expr, self.features, self.target)
        return self.input_expr, self.input_expr

    # ── Deferred exprs ─────────────────────────────────────────

    @property
    def deferred_exprs(self):
        """Build deferred exprs for every pipeline via build_exprs_fn.

        Returns dict: {pipeline_name: {expr_name: ibis_expr, ...}, ...}

        The build_exprs_fn receives the raw sklearn pipeline, train/test
        exprs, features, target, and pred_name — so the example controls
        (and shows) every step: Pipeline.from_instance, fit, predict, metrics.
        """
        if not self.build_exprs_fn:
            return {}
        train_expr, test_expr = self.xorq_split
        return {
            name: self.build_exprs_fn(
                self.pipelines[name],
                train_expr,
                test_expr,
                self.features,
                self.target,
                f"{self.pred_col}_{name}",
            )
            for name in self.names
        }

    # ── Compute ────────────────────────────────────────────────

    def compute_with_sklearn(self):
        """Call the example's sklearn compute function."""
        if self.compute_sklearn_fn:
            return self.compute_sklearn_fn(self)

    def compute_with_xorq(self):
        """Call the example's xorq compute function."""
        if self.compute_xorq_fn:
            return self.compute_xorq_fn(self)

    # ── Assertions ────────────────────────────────────────────

    def assert_values(self, sk_results, xo_results, atol=1e-3):
        """Build assertion DataFrames via build_assertions_fn, then assert equality.

        Delegates to the example's ``build_assertions_fn`` to construct
        ``(label, sk_df, xo_df)`` tuples, then asserts each pair matches.
        """
        if not self.build_assertions_fn:
            return
        assertions = self.build_assertions_fn(sk_results, xo_results, self)
        print("\n=== Assertions ===")
        for label, sk_df, xo_df in assertions:
            pd.testing.assert_frame_equal(sk_df, xo_df, atol=atol)
            print(f"{label} match (atol={atol}).")

    # ── Plot ───────────────────────────────────────────────────

    def plot(self, sk_results, xo_results):
        """Call the example's plot function."""
        if self.plot_fn:
            return self.plot_fn(sk_results, xo_results, self)
