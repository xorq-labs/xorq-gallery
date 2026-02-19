"""Gallery utilities.

TODO: Sequential (temporal) splitting should be supported directly by
xorq's train_test_splits, which currently only does hash-based splitting.
Adding an order_by + shuffle=False mode is an easy lift.  This helper
bridges the gap while we get the gallery together.
"""

from sklearn.model_selection import TimeSeriesSplit
from xorq.expr.ml.cross_validation import (
    _fold_pairs_from_fold_expr,
    _make_folds_from_sklearn,
)


def deferred_sequential_split(expr, *, features, target, order_by, test_size=0.3333):
    """Split an ibis expression into (train, test) preserving row order.

    Uses TimeSeriesSplit(n_splits=2) under the hood so the last fold gives
    roughly the first 2/3 for training and the last 1/3 for testing -- the
    deferred equivalent of ``train_test_split(shuffle=False, test_size=0.3333)``.

    Parameters
    ----------
    expr : ibis expression
        The table to split.
    features : tuple[str, ...]
        Feature column names.
    target : str
        Target column name.
    order_by : str
        Column that defines the temporal / sequential order.
    test_size : float, optional
        Ignored for now (always 1/3). Kept for forward-compatibility with a
        future ``train_test_splits(shuffle=False)`` implementation.

    Returns
    -------
    train_expr, test_expr : tuple of ibis expressions
    """
    cv = TimeSeriesSplit(n_splits=2)
    fold_expr = _make_folds_from_sklearn(
        expr=expr,
        cv=cv,
        features=features,
        target=target,
        order_by=order_by,
    )
    fold_pairs = _fold_pairs_from_fold_expr(fold_expr, n_splits=2)
    return fold_pairs[-1]  # last fold = largest train set
