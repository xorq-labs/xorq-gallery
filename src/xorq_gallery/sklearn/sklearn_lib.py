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
    load_data = field(validator=is_callable())
    split_data = field(validator=is_callable())
    #
    make_sklearn_result = field(validator=is_callable())
    make_deferred_xorq_result = field(validator=is_callable())
    make_xorq_result = field(validator=is_callable())
    compare_results_fn = field(validator=is_callable())
    plot_results_fn = field(validator=is_callable())

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
        return self.compare_results_fn(self.sklearn_results, self.xorq_results)

    def plot_results(self):
        return self.plot_result_fn(self.sklearn_results, self.xorq_results)
