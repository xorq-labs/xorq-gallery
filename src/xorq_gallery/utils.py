"""Gallery utilities.

TODO: Sequential (temporal) splitting should be supported directly by
xorq's train_test_splits, which currently only does hash-based splitting.
Adding an order_by + shuffle=False mode is an easy lift.  This helper
bridges the gap while we get the gallery together.
"""

from io import BytesIO
from pathlib import Path

from toolz import compose


def deferred_sequential_split(expr, *, features, target, order_by):
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
    from sklearn.model_selection import TimeSeriesSplit
    from xorq.expr.ml.cross_validation import (
        _fold_pairs_from_fold_expr,
        _make_folds_from_sklearn,
    )

    cv = TimeSeriesSplit(n_splits=2)
    fold_expr = _make_folds_from_sklearn(
        expr=expr,
        cv=cv,
        features=features,
        target=target,
        order_by=order_by,
    )
    # last fold = largest train set
    (*_, fold_pair) = _fold_pairs_from_fold_expr(fold_expr, n_splits=2)
    return fold_pair


# ---------------------------------------------------------------------------
# Deferred matplotlib plotting via UDAF
# ---------------------------------------------------------------------------


def _fig_to_png_bytes(fig, close=True):
    """Serialize a matplotlib Figure to PNG bytes."""
    import matplotlib.pyplot as plt

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    if close:
        plt.close(fig)
    return buf.getvalue()


def deferred_matplotlib_plot(expr, fn, name="plot"):
    """Create a deferred plot that materializes on execute.

    Wraps a plotting function in a pandas DataFrame UDAF.  The function
    receives the full DataFrame and must return a ``matplotlib.figure.Figure``.
    The figure is serialized to PNG bytes and stored in a binary column.

    Parameters
    ----------
    expr : ibis expression
        The table whose data the plotting function needs.
    fn : callable(pd.DataFrame) -> matplotlib.figure.Figure
        Plotting function.
    name : str, optional
        Column name for the binary output (default ``"plot"``).

    Returns
    -------
    ibis scalar expression
        Scalar expression whose ``.execute()`` returns PNG bytes.
    """

    import xorq.expr.datatypes as dt
    from xorq.expr.udf import agg

    plot_udaf = agg.pandas_df(
        fn=compose(_fig_to_png_bytes, fn),
        schema=expr.schema(),
        return_type=dt.binary,
        name=name,
    )
    return plot_udaf.on_expr(expr)


def save_plot(img_bytes, path):
    """Write a deferred plot result to disk.

    Parameters
    ----------
    img_bytes : bytes
        PNG bytes from ``deferred_matplotlib_plot(...).execute()``.
    path : str
        File path to write the PNG to.
    """
    Path(path).write_bytes(img_bytes)
    print(f"Plot saved to {path}")


def show_plot(img_bytes):
    """Display a deferred plot result.

    Parameters
    ----------
    img_bytes : bytes
        PNG bytes from ``deferred_matplotlib_plot(...).execute()``.
    """
    import matplotlib.pyplot as plt

    img = load_plot_bytes(img_bytes)
    plt.imshow(img)
    plt.axis("off")
    plt.show(block=False)


def fig_to_image(fig, close=True):
    """Render a matplotlib Figure to a numpy RGBA image array.

    Handles HiDPI/retina displays correctly by reading actual buffer
    dimensions rather than logical dimensions.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to render.

    Returns
    -------
    numpy.ndarray
        RGBA image array suitable for ``ax.imshow()``.
        The figure is closed after rendering.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    if close:
        plt.close(fig)
    return img


def load_plot_bytes(img_bytes):
    """Load PNG bytes as a numpy image array for embedding in subplots.

    Parameters
    ----------
    img_bytes : bytes
        PNG bytes from ``deferred_matplotlib_plot(...).execute()``.

    Returns
    -------
    numpy.ndarray
        Image array suitable for ``ax.imshow()``.
    """
    import matplotlib.image as mpimg

    return mpimg.imread(BytesIO(img_bytes), format="png")


load_plot = compose(Path.read_bytes, Path)
