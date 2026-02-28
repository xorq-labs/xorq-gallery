from functools import cache

import pandas as pd
import xorq.api as xo
from attr.validators import (
    deep_iterable,
    instance_of,
    is_callable,
)
from attrs import (
    field,
    frozen,
)
from sklearn.base import clone
from toolz import curry
from xorq.common.utils.func_utils import return_constant
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline


def split_data_nop(df):
    return (df, df)


@curry
def make_sklearn_result(
    pipeline,
    train_data,
    test_data,
    features,
    target,
    metrics_names_funcs,
    make_other=return_constant(None),
):
    ((X_train, y_train), (X_test, y_test)) = (
        (df[list(features)], df[target]) for df in (train_data, test_data)
    )
    fitted = clone(pipeline).fit(X_train, y_train)
    preds = fitted.predict(X_test)
    metrics = {
        metric_name: metric_func(y_test, preds)
        for metric_name, metric_func in metrics_names_funcs
    }
    other = make_other(fitted)
    result = {
        "fitted": fitted.steps[-1][-1],
        "preds": preds,
        "metrics": metrics,
    } | ({"other": other} if other else {})
    return result


@curry
def make_deferred_xorq_result(
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
    preds = xorq_fitted.predict(test_data, name=pred)
    metrics = {
        name: preds.agg(
            **{
                name: deferred_sklearn_metric(
                    target=target, pred=pred, metric=metric_fn
                )
            }
        )
        for name, metric_fn in metrics_names_funcs
    }
    other = make_other(xorq_fitted)
    deferred_xorq_result = {
        "xorq_fitted": xorq_fitted,
        "preds": preds,
        "metrics": metrics,
        "other": other if other else {},
    }
    return deferred_xorq_result


def make_xorq_result(deferred_xorq_result):
    xorq_fitted, preds, metrics, other = (
        deferred_xorq_result[name]
        for name in (
            "xorq_fitted",
            "preds",
            "metrics",
            "other",
        )
    )
    other = {k: v() for k, v in other.items()}
    result = {
        "fitted": xorq_fitted.fitted_steps[-1].model,
        "preds": preds.execute(),
        "metrics": {name: expr.as_scalar().execute() for name, expr in metrics.items()},
    } | ({"other": other} if other else {})
    return result


@frozen
class SklearnXorqComparator:
    names_pipelines = field(
        validator=deep_iterable(instance_of(tuple), instance_of(tuple))
    )
    features = field(validator=deep_iterable(instance_of(str), instance_of(tuple)))
    target = field(validator=instance_of(str))
    pred = field(validator=instance_of(str))
    metrics_names_funcs = field(
        validator=deep_iterable(instance_of(tuple), instance_of(tuple)),
    )
    #
    load_data = field(validator=is_callable())
    split_data = field(validator=is_callable())
    compare_results_fn = field(validator=is_callable())
    plot_results_fn = field(validator=is_callable())
    #
    make_sklearn_result = field(validator=is_callable(), default=make_sklearn_result)
    make_deferred_xorq_result = field(
        validator=is_callable(), default=make_deferred_xorq_result
    )
    make_xorq_result = field(validator=is_callable(), default=make_xorq_result)

    @property
    @cache
    def df(self):
        return self.load_data()

    def get_split_data(self):
        # DataFrame.attrs can cause issues, copy removes them
        df = self.df.pipe(lambda t: pd.DataFrame(t, columns=t.columns))
        return self.split_data(df)

    @property
    @cache
    def sklearn_results(self):
        train, test = self.get_split_data()
        results = {
            name: self.make_sklearn_result(
                pipeline,
                train,
                test,
                self.features,
                self.target,
                self.metrics_names_funcs,
            )
            for name, pipeline in self.names_pipelines
        }
        return results

    @property
    @cache
    def deferred_xorq_results(self):
        train, test = (xo.memtable(el) for el in self.get_split_data())
        results = {
            name: self.make_deferred_xorq_result(
                pipeline,
                train,
                test,
                self.features,
                self.target,
                self.metrics_names_funcs,
                self.pred,
            )
            for name, pipeline in self.names_pipelines
        }
        return results

    @property
    @cache
    def xorq_results(self):
        results = {
            name: self.make_xorq_result(deferred_xorq_result)
            for name, deferred_xorq_result in self.deferred_xorq_results.items()
        }
        return results

    @property
    def result_comparison(self):
        return self.compare_results_fn(self)

    def plot_results(self):
        return self.plot_results_fn(self)
