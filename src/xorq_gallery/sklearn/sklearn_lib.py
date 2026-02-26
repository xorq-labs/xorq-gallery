from functools import cache

from attr.validators import (
    deep_iterable,
    instance_of,
)
from attrs import (
    field,
    frozen,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from xorq.api import Expr
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline


@frozen
class SklearnXorqComparator:
    sklearn_pipeline = field(validator=instance_of(SklearnPipeline))
    input_expr = field(validator=instance_of(Expr))
    kwargs_tuple = field(
        validator=deep_iterable(instance_of(tuple), instance_of(tuple))
    )
    metrics_names_funcs = field(
        validator=deep_iterable(instance_of(tuple), instance_of(tuple)), default=()
    )

    @property
    @cache
    def df(self):
        return self.input_expr.execute()

    @property
    def X(self):
        return self.df[list(self.kwargs["features"])].values

    @property
    def y(self):
        return self.df[self.kwargs["target"]].values

    @property
    def kwargs(self):
        return dict(self.kwargs_tuple)

    @property
    def xorq_pipeline(self):
        return Pipeline.from_instance(self.sklearn_pipeline)

    @property
    def fitted_xorq_pipeline(self):
        return self.xorq_pipeline.fit(
            self.input_expr,
            features=self.kwargs["features"],
            target=self.kwargs["target"],
        )

    @property
    def xorq_prediction(self):
        return self.fitted_xorq_pipeline.predict(
            self.input_expr, name=self.kwargs["pred"]
        )

    @property
    @cache
    def fitted_sklearn_pipeline(self):
        instance = self.xorq_pipeline.instance
        return instance.fit(self.X, self.y)

    @property
    def sklearn_prediction(self):
        return self.fitted_sklearn_pipeline.predict(self.X)

    @property
    def sklearn_metrics(self):
        return {
            name: metric(self.y, self.sklearn_prediction)
            for name, metric in self.metrics_names_funcs
        }

    @property
    def xorq_metrics(self):
        return self.xorq_prediction.agg(
            **{
                name: deferred_sklearn_metric(
                    target=self.kwargs["target"],
                    pred=self.kwargs["pred"],
                    metric=metric,
                )
                for name, metric in self.metrics_names_funcs
            }
        )
