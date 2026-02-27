"""Shared library for sklearn gallery examples.

SklearnXorqComparator is a frozen attrs data bag that holds the declaration
(pipelines, data, column config) and builds the deferred xorq expression
graph.  All other behavior -- computing, asserting, plotting -- lives in
conventionally-named functions defined inline in the example file.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import xorq.api as xo
from attr.validators import deep_iterable, instance_of, is_callable, optional
from attrs import field, frozen


@frozen
class SklearnXorqComparator:
    """Data bag for an sklearn-vs-xorq gallery example.

    Holds the declaration (what pipelines, what data, what columns) and
    builds the deferred xorq expression graph via ``build_exprs_fn``.

    Everything else -- eager sklearn computation, deferred xorq execution,
    assertions, plotting -- is done by conventionally-named functions in the
    example file.  The comparator is passed around for its data accessors,
    not as an orchestrator.

    Parameters
    ----------
    name : str
        Example name. Also used as the ibis table name for registration.
    named_pipelines : tuple[tuple[str, Any], ...]
        Named sklearn pipelines to compare. Always use explicit named steps.
        Tuple-of-tuples for hashability on frozen class.
    df : pd.DataFrame
        The dataset. ``_load_data()`` is called inline in the example so the
        reader sees the data source.
    features : tuple[str, ...]
        Feature column names.
    build_exprs_fn : callable
        ``(sklearn_pipeline, input_expr, features, target, pred_name)``
        ``-> dict[str, Expr]``.
        Defined in the example. Receives the raw sklearn pipeline and the
        unsplit ``input_expr``.  The function handles splitting (if any) as
        part of the expression graph, then does ``Pipeline.from_instance``,
        fit, predict, and deferred metrics.
        Returns dict with keys like ``"fitted_pipeline"``, ``"preds"``,
        ``"metrics"``.
    target : str or None
        Target column name. None for unsupervised examples.
    pred_col : str
        Base prediction column name (default ``"pred"``).
    metrics : tuple[tuple[str, Callable], ...]
        Named metric functions, e.g. ``(("r2", r2_score),)``.
        Tuple-of-tuples for hashability.
    """

    name: str = field(validator=instance_of(str))
    named_pipelines: tuple = field(
        validator=deep_iterable(instance_of(tuple), instance_of(tuple))
    )
    df: pd.DataFrame = field(validator=instance_of(pd.DataFrame), eq=False, hash=False)
    features: tuple = field(validator=instance_of(tuple))
    build_exprs_fn: Callable = field(validator=is_callable(), eq=False, hash=False)
    target: str | None = field(validator=optional(instance_of(str)), default=None)
    pred_col: str = field(validator=instance_of(str), default="pred")
    metrics: tuple = field(
        validator=deep_iterable(instance_of(tuple), instance_of(tuple)),
        default=(),
    )

    # -- Data access --------------------------------------------------------

    @property
    def pipelines(self):
        """Pipeline dict from the tuple-of-tuples."""
        return dict(self.named_pipelines)

    @property
    def input_expr(self):
        """Register df as an ibis table. Cheap -- builds expr tree, no execution."""
        return xo.connect().register(self.df, self.name)

    @property
    def names(self):
        """Pipeline names as a tuple."""
        return tuple(name for name, _ in self.named_pipelines)

    # -- Deferred exprs -----------------------------------------------------

    @property
    def deferred_exprs(self):
        """Build deferred exprs for every pipeline via build_exprs_fn.

        Returns dict: ``{pipeline_name: {expr_name: ibis_expr, ...}, ...}``

        The ``build_exprs_fn`` receives the raw sklearn pipeline and the
        unsplit ``input_expr`` -- so the example controls (and shows) every
        step: splitting, ``Pipeline.from_instance``, fit, predict, metrics.
        """
        return {
            name: self.build_exprs_fn(
                self.pipelines[name],
                self.input_expr,
                self.features,
                self.target,
                f"{self.pred_col}_{name}",
            )
            for name in self.names
        }


def assert_results(assertions, atol=1e-3):
    """Assert that sklearn and xorq results match.

    Parameters
    ----------
    assertions : list[tuple[str, DataFrame, DataFrame]]
        ``(label, sk_df, xo_df)`` tuples with informative column names
        so failures pinpoint the source.
    atol : float
        Absolute tolerance for ``assert_frame_equal``.
    """
    print("\n=== Assertions ===")
    for label, sk_df, xo_df in assertions:
        pd.testing.assert_frame_equal(sk_df, xo_df, atol=atol)
        print(f"  {label} match (atol={atol}).")
