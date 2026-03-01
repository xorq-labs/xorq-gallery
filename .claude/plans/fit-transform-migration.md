# Plan: fit-transform migration for preprocessing and decomposition scripts

## Scope

Three scripts use `fit_transform` / `fit` + `transform` with no `predict()`:

| Script | Models | Feature variation |
|---|---|---|
| `preprocessing/plot_all_scaling.py` | 9 scalers (+ identity) | Single `FEATURE_COLS = ("MedInc", "AveOccup")` |
| `preprocessing/plot_discretization_strategies.py` | 3 strategies × 3 datasets | Single `FEATURE_COLS = ("x0", "x1")` per dataset |
| `decomposition/plot_faces_decomposition.py` | 8 decompositions | Two feature sets: `pixel_cols` (NMF) vs `centered_cols` (all others) |

---

## Part 1: additions to `sklearn_lib.py`

Three new functions mirroring the default `make_*_result` trio but substituting `fit_transform` / `transform` for `predict`. The signatures are identical so they drop straight into the `make_sklearn_result`, `make_deferred_xorq_result`, `make_xorq_result` fields of `SklearnXorqComparator`.

### `make_sklearn_fit_transform_result`

```python
@curry
def make_sklearn_fit_transform_result(
    pipeline,
    train_data,
    test_data,
    features,
    target,             # accepted but ignored
    metrics_names_funcs,  # accepted but ignored (always empty for transform-only)
    make_other=return_constant(None),
):
    X_train = train_data[list(features)]
    X_test  = test_data[list(features)]
    fitted  = clone(pipeline).fit(X_train)
    transformed = fitted.transform(X_test)
    other = make_other(fitted)
    return {
        "fitted":      fitted.steps[-1][-1],
        "transformed": transformed,
        "metrics":     {},
    } | ({"other": other} if other else {})
```

### `make_deferred_xorq_fit_transform_result`

```python
@curry
def make_deferred_xorq_fit_transform_result(
    pipeline,
    train_data,
    test_data,
    features,
    target,             # accepted but ignored — passed as None to Pipeline.fit
    metrics_names_funcs,
    pred,               # accepted but ignored
    make_other=return_constant(None),
):
    xorq_fitted = Pipeline.from_instance(pipeline).fit(
        train_data, features=features, target=None
    )
    transformed = xorq_fitted.transform(test_data)  # ibis expression
    other = make_other(xorq_fitted)
    return {
        "xorq_fitted": xorq_fitted,
        "transformed": transformed,
        "metrics":     {},
        "other":       other if other else {},
    }
```

### `make_xorq_fit_transform_result`

```python
def make_xorq_fit_transform_result(deferred_xorq_result):
    xorq_fitted = deferred_xorq_result["xorq_fitted"]
    transformed = deferred_xorq_result["transformed"]
    other = deferred_xorq_result.get("other", {})
    other = {k: v() for k, v in other.items()}
    return {
        "fitted":      xorq_fitted.fitted_steps[-1].model,
        "transformed": transformed.execute(),
        "metrics":     {},
    } | ({"other": other} if other else {})
```

**No changes needed to `SklearnXorqComparator` itself.** The `target` validator (`instance_of(str)`) is satisfied by keeping a real (or dummy) target column in the DataFrame — the custom `make_*` functions simply ignore it. The `pred` field is similarly passed through but unused.

---

## Part 2: `plot_all_scaling.py`

**Current:** `sklearn_way()` / `xorq_way()` pattern, no comparator, no module-level xorq exprs.

**Changes:**

1. `load_data()` — returns flat DataFrame: `MedInc`, `AveOccup`, plus `y_scaled` (target, `minmax_scale(y_full)`) as a real column (not just a dict value). Passes `df.attrs["y_full_range"] = (y_full.min(), y_full.max())` for the colorbar.

2. Drop the "Unscaled data" / `None` scaler entry from `names_pipelines` — it needs no fitting. Handle it separately in `compare_results_fn` and `plot_results_fn` as the baseline (`comparator.df[list(FEATURE_COLS)].values`).

3. `split_data = split_data_nop`.

4. `metrics_names_funcs = ()`.

5. `compare_results_fn`: compare mean/std of each scaler's transformed output between sklearn and xorq (already done via `pd.testing.assert_frame_equal`).

6. `plot_results_fn`: uses `sklearn_results[name]["transformed"]` and `xorq_results[name]["transformed"]` directly (both already numpy arrays / DataFrames after execution). No `deferred_matplotlib_plot` needed.

7. Module-level overrides:
   ```python
   make_sklearn_result = make_sklearn_fit_transform_result
   make_deferred_xorq_result = make_deferred_xorq_fit_transform_result
   make_xorq_result = make_xorq_fit_transform_result
   ```

8. Module-level exposure:
   ```python
   (xorq_standard_transformed, xorq_minmax_transformed, ...) = (
       comparator.deferred_xorq_results[name]["transformed"] for name in methods
   )
   ```

**`TARGET_COL = "y_scaled"`, `FEATURE_COLS = ("MedInc", "AveOccup")`, `PRED_COL = "pred"` (unused).**

---

## Part 3: `plot_discretization_strategies.py`

**Current:** `sklearn_way()` / `xorq_way()` per dataset, transforms a meshgrid for contour visualization.

**Key tension:** the contour plot needs predictions on a 300×300 meshgrid derived from each dataset's bounds — not on the training data. This is the same "fine grid" pattern used in `tree/plot_tree_regression.py`.

**Solution:** `split_data_nop` (train == test) so the comparator fits correctly, then in `plot_results_fn` call `result["fitted"].transform(meshgrid)` (sklearn) and `deferred_xorq_results[name]["xorq_fitted"].transform(xo.memtable(grid_df)).execute()` (xorq) — exactly the same pattern as tree_regression.

**Changes:**

1. `load_data()` — returns a single combined DataFrame (all 3 datasets stacked) with a `dataset_id` column (0/1/2). `TARGET_COL = "dataset_id"` (real column, ignored by fit-transform). `FEATURE_COLS = ("x0", "x1")`.

2. One `comparator` per dataset (3 total), or alternatively one comparator with `names_pipelines = ((f"ds{i}_{strategy}", pipeline), ...)` for all 9 combinations. **Recommended: one comparator per dataset** — keeps `plot_results_fn` clean.

3. `split_data = split_data_nop`. `metrics_names_funcs = ()`.

4. `compare_results_fn`: compares discretized bin assignments on training data between sklearn and xorq for each strategy.

5. `plot_results_fn`: builds the 4-panel figure (input + 3 strategies) per dataset. Meshgrid computed from `comparator.df` bounds; contour values from `result["fitted"].transform(grid)` and `xorq_fitted.transform(xo.memtable(grid_df)).execute()`.

6. Module-level exposure (one per strategy per dataset):
   ```python
   (xorq_ds0_uniform_transformed, ...) = (
       comparator_ds0.deferred_xorq_results[name]["transformed"] for name in methods
   )
   ```

---

## Part 4: `plot_faces_decomposition.py`

**Current:** `sklearn_way()` / `xorq_way()` pattern, two feature sets (`pixel_cols` for NMF, `centered_cols` for all others), components_ extracted after `transform()`.

**Key issue:** the split `pixel_cols` vs `centered_cols` per model means a single `FEATURE_COLS` doesn't cover all models. **Solution: two comparators.**

1. `comparator_pixel` — `FEATURE_COLS = pixel_cols`, `names_pipelines = ((NMF, nmf_pipeline),)`.
2. `comparator_centered` — `FEATURE_COLS = centered_cols`, `names_pipelines = ((PCA, ...), (ICA, ...), (SPARSE_PCA, ...), (DICT_LEARNING, ...), (FA, ...), (DICT_POS, ...))`.

**MiniBatchKMeans** is a clusterer (no `transform()` method). Keep it handled eagerly outside the comparator — fit it in `compare_results_fn` or `main()` using `comparator_centered.sklearn_results` and `comparator_centered.xorq_results` for context.

**Changes:**

1. `load_data()` — returns DataFrame with `pixel_cols + centered_cols + ["subject_id"]`. `TARGET_COL = "subject_id"` (ignored). Two `FEATURE_COLS` constants.

2. `make_sklearn_other` / `make_xorq_other` — extract `components_` (or `cluster_centers_`) from the fitted model:
   ```python
   def make_sklearn_other(fitted):
       return {"components": fitted.components_[:N_COMPONENTS].copy()}
   def make_xorq_other(xorq_fitted):
       return {"components": lambda: xorq_fitted.fitted_steps[-1].model.components_[:N_COMPONENTS].copy()}
   ```
   Pass these as curried `make_other` arg to `make_sklearn_fit_transform_result` / `make_deferred_xorq_fit_transform_result`.

3. `compare_results_fn`: reads `result["other"]["components"]` from both comparators, prints max/mean difference.

4. `plot_results_fn`: reads `result["other"]["components"]` to build gallery figures.

5. Module-level exposure (`comparator_pixel` + `comparator_centered` both contribute `["transformed"]` exprs). Only the `centered` comparator's transforms are meaningful for the xorq build (pixel NMF optional).

---

## Implementation order

1. `sklearn_lib.py` — add the three `make_*_fit_transform_*` functions (prerequisite for all three scripts)
2. `plot_all_scaling.py` — simplest (single comparator, single FEATURE_COLS, no fine-grid)
3. `plot_discretization_strategies.py` — medium (three comparators, fine-grid transform)
4. `plot_faces_decomposition.py` — hardest (two comparators, components_ extraction, KMeans special case)

---

## Cross-references with `.claude/sklearn-comparator-migration.md`

### 1. Two decision table entries become stale after implementation

- `"fit_transform pipelines (scalers, discretizers) → Skip"` — replace with guidance to use the `make_sklearn_fit_transform_result` trio and `metrics_names_funcs = ()`.
- `"Unsupervised (NMF/LDA, no target column, no .predict()) → Skip"` — replace with guidance to use two comparators split by feature set, `make_other` for `components_`, and handle non-transformer models (KMeans) eagerly outside the comparator.

Update both entries in the migration table after all three scripts are done.

### 2. Standard steps step 9 changes for fit-transform scripts

The standard checklist says:
> Expose deferred preds at module level: `(xorq_foo_preds, ...) = (comparator.deferred_xorq_results[name]["preds"] for name in methods)`

For fit-transform scripts the key is `"transformed"` not `"preds"`:
```python
(xorq_foo_transformed, ...) = (
    comparator.deferred_xorq_results[name]["transformed"] for name in methods
)
```

### 3. Identity/passthrough pipeline has no decision table entry

`plot_all_scaling.py` has an "Unscaled data" entry (no scaler, `None`). Plan: drop it from `names_pipelines` and reference `comparator.df[list(FEATURE_COLS)].values` directly in compare/plot callbacks as the baseline. Add a decision table row:
> "One `names_pipelines` entry is a passthrough/identity (no fitting)" → drop it from `names_pipelines`; handle as `comparator.df[list(FEATURE_COLS)]` baseline in callbacks.

### 4. Two comparators for different feature sets vs. different datasets

The existing entry "Multiple datasets × multiple models → one comparator per dataset" does not cover the faces case where the split is by *feature set*, not by dataset. Add a decision table row:
> "Multiple models require different `FEATURE_COLS` (e.g. NMF on raw pixels, PCA on centered pixels)" → one comparator per feature set; share the same `load_data()` and `split_data`.

### 5. Stochastic transformer: approximate equality, not exact match

`QuantileTransformer` uses random subsampling internally, so sklearn and xorq outputs diverge slightly due to row-ordering differences. The existing code already uses `pd.testing.assert_frame_equal(rtol=0.05, atol=0.01)`. Add a decision table row:
> "Transformer uses random subsampling (e.g. `QuantileTransformer`)" → use `pd.testing.assert_frame_equal(rtol=..., atol=...)` in `compare_results_fn` rather than `np.testing.assert_allclose`; document the tolerance and its reason in a comment.
