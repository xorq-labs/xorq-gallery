## xorq-gallery Example Conventions

When working on sklearn gallery examples, follow these patterns.

### Design Principle

Structure comes from **naming convention**, not from the framework. The `SklearnXorqComparator` class holds only what is truly immutable across all examples (data declaration + deferred expression graph). Everything else -- computing, asserting, plotting -- lives as conventionally-named functions in the example file.

`compute_sklearn` is pure sklearn with no xorq imports. `compute_xorq` materializes deferred expressions. A reader should be able to understand each side independently.

### SklearnXorqComparator

A frozen attrs **data bag** from `xorq_gallery.sklearn.sklearn_lib`. It holds the declaration (pipelines, data, columns) and builds the deferred xorq expression graph. It is **not** an orchestrator -- it has no compute, assert, or plot methods.

**Fields:**
- `name` -- example name, also the ibis table name
- `named_pipelines` -- `tuple[tuple[str, SklearnPipeline], ...]`
- `df` -- the dataset as a pandas DataFrame
- `features` -- tuple of feature column names
- `target` -- target column name (or None for unsupervised)
- `pred_col` -- base prediction column name (default `"pred"`)
- `metrics` -- `tuple[tuple[str, callable], ...]` for hashability
- `build_exprs_fn` -- the one injectable callable (see below)

**Properties:**
- `input_expr` -- `xo.connect().register(df, name)`
- `names` -- pipeline name tuple
- `pipelines` -- dict from `named_pipelines`
- `deferred_exprs` -- iterates pipelines, calls `build_exprs_fn` for each

### File Layout (in order)

1. Module docstring (sklearn paragraph, xorq paragraph, dataset)
2. Constants (`N_SAMPLES`, `FEATURE_COLS`, `TARGET_COL`, `ROW_IDX`, `PRED_COL`, etc.)
3. `_load_data()` -- returns a `pd.DataFrame`
4. Plot helpers (private functions for building matplotlib figures)
5. `build_exprs(...)` -- the full xorq expression chain including split
6. `compute_sklearn(df, pipelines, ...)` -- pure sklearn, explicit args
7. `compute_xorq(deferred_exprs, ...)` -- materializes deferred expressions
8. `build_assertions(sk, xo)` -- returns `[(label, sk_df, xo_df), ...]`
9. `plot(sk, xo, ...)` -- composite figure
10. `SHARED_SKLEARN_PIPELINES` -- tuple-of-tuples
11. Comparator construction
12. Module-level exprs for `xorq build --expr`
13. `main()` and `if __name__` guard

### Convention Function Signatures

These functions are defined in every example file. Signatures vary per example because the functions take explicit arguments (not a comparator), but the names are consistent.

**`_load_data()`**
`() -> pd.DataFrame`. Always present. Reader sees where data comes from.

**`build_exprs(sklearn_pipeline, input_expr, features, target, pred_name)`**
`-> dict[str, Expr]`. The full xorq expression chain. Split is here because it is part of the expression graph. Returns dict with `"fitted_pipeline"`, `"preds"`, `"metrics"`. Zero `.execute()` calls. This is the one function also passed to the comparator as `build_exprs_fn`.

**`compute_sklearn(df, pipelines, ...)`**
`-> results_dict`. Pure sklearn -- no xorq imports, no comparator. Takes the DataFrame, the pipeline tuples, and whatever else it needs (target_col, etc.) as explicit arguments. Does its own split if needed. A data scientist should be able to read this without knowing what xorq is.

**`compute_xorq(deferred_exprs, ...)`**
`-> results_dict`. Takes `deferred_exprs` dict (from `comparator.deferred_exprs`) and any extra arguments it needs. Calls `.execute()` on the deferred expressions. This is where deferred becomes eager.

**`build_assertions(sk, xo)`**
`-> list[tuple[str, DataFrame, DataFrame]]`. Takes the two result dicts directly. Column names should be informative so failures pinpoint the source.

**`plot(sk, xo, ...)`**
`-> None`. Takes results and whatever data it needs for the figure. Single composite output (sklearn left, xorq right or vertically stacked). Uses `fig_to_image()` helper.

**`main()`**
Orchestrates: `compute_sklearn -> compute_xorq -> assert_results(build_assertions(...)) -> plot`. This is the only place the comparator is used to wire things together (via `comparator.deferred_exprs`, `comparator.df`, `comparator.target`).

### Hard Rules

**Never use `make_pipeline`**
Always use explicit `SklearnPipeline([("step_name", step)])` with named steps. `Pipeline.from_instance` requires named steps.

**Named pipelines are tuple-of-tuples**
`SHARED_SKLEARN_PIPELINES` must be a tuple of `(name, SklearnPipeline)` tuples for hashability on the frozen class. Never use a dict.

**`compute_sklearn` has no xorq imports and no comparator argument**
It takes `df`, `pipelines`, and whatever else it needs as plain arguments. It does its own split. It uses `clone()` from sklearn to get fresh pipeline copies.

**`compute_xorq` takes `deferred_exprs`, not the comparator**
It receives a plain dict and calls `.execute()` on expressions.

**`build_exprs` owns the split on the xorq side**
The split (filter, `train_test_splits`, `deferred_sequential_split`, or nothing) is part of the expression graph. It lives inside `build_exprs`, not on the class.

**`compute_sklearn` does its own split**
The sklearn side does its own equivalent split so the reader sees both strategies side by side. Sometimes this means `apply_deterministic_sort(table).execute()` to get a DataFrame matching DataFusion's row order.

**`build_assertions` and `plot` take explicit arguments, not the comparator**
Pass `sk`, `xo`, `df`, `target_col`, etc. directly.

**`assert_results` is a shared helper**
Import from `xorq_gallery.sklearn.sklearn_lib`. It takes the list from `build_assertions` and calls `pd.testing.assert_frame_equal` on each pair.

### Train/Test Splits

**Splits must produce identical partitions**
The #1 cause of sklearn-vs-xorq divergence is mismatched splits. Always verify that both paths see the same train and test rows.

**The xorq split is part of the expression graph**
It lives inside `build_exprs` because it is a node in the ibis expression tree:
```python
def build_exprs(sklearn_pipeline, input_expr, features, target, pred_name):
    train_expr = input_expr.filter(input_expr[ROW_IDX] < CUTOFF)
    test_expr = input_expr.filter(input_expr[ROW_IDX] >= CUTOFF)
    pipeline = Pipeline.from_instance(sklearn_pipeline)
    fitted = pipeline.fit(train_expr, features=features, target=target)
    preds = fitted.predict(test_expr, name=pred_name)
    ...
```

**The sklearn split is in `compute_sklearn`**
Eager pandas/numpy slicing, visible to the reader:
```python
def compute_sklearn(df, pipelines):
    train_df = df[df[ROW_IDX] < CUTOFF]
    test_df = df[df[ROW_IDX] >= CUTOFF]
    ...
```

**`deferred_sequential_split` does NOT exactly match `train_test_split`**
For exact match, use a row-index cutoff (see above).

**Hash-based splits** use `xo.train_test_splits` with `unique_key=ROW_IDX` inside `build_exprs`. Materialize to pandas for sklearn in `compute_sklearn`.

**No split needed?** `build_exprs` uses `input_expr` for both train and test. `compute_sklearn` uses the full DataFrame.

### Multiple Comparators

When the same pipelines apply to different targets (e.g., quantile regression with normal vs pareto noise), create a `comparators` dict:
```python
comparators = {
    "normal": SklearnXorqComparator(name="quantile_normal", target="y_normal", ...),
    "pareto": SklearnXorqComparator(name="quantile_pareto", target="y_pareto", ...),
}
```
The `build_exprs_fn` can be a closure when it needs the target column:
```python
build_exprs_fn=build_exprs_with_metrics("y_normal"),
```

### Module-level Exprs for `xorq build --expr`

Every example must expose all deferred expressions at module level:
```python
xorq_exprs = comparator.deferred_exprs

xorq_lasso_fitted_pipeline = xorq_exprs["Lasso"]["fitted_pipeline"]
xorq_lasso_preds = xorq_exprs["Lasso"]["preds"]
xorq_lasso_metrics = xorq_exprs["Lasso"]["metrics"]
```

### Deferred Plot Pitfalls

**Data fed to `deferred_matplotlib_plot` must match what the sklearn plot uses**
The deferred plot function receives a DataFrame via the UDAF. Mismatched data is the #1 cause of visual bugs.

**Curried plot function args must be hashable**
All curried keyword args must be hashable (tuples, strings, numbers, frozen containers). Common fixes:
- Convert numpy arrays to `freeze(arr.tolist())`
- Convert numpy `target_names` to `tuple(class_names)`
- **Never pass sklearn estimator objects** as curry kwargs

### Accessing Fitted Model Attributes

To get sklearn model attributes (e.g., `coef_`) from a fitted xorq pipeline:
```python
coef = fitted.fitted_steps[-1].model.coef_
```

### Deterministic Ordering Fallback

For order-sensitive algorithms, use `apply_deterministic_sort`:
```python
from xorq.expr.ml.cross_validation import apply_deterministic_sort

con = xo.connect()
raw_table = con.register(df, "table_name")
table = apply_deterministic_sort(raw_table)
df = table.execute()  # materialise sorted order back to pandas
```
