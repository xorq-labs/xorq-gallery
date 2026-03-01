# Plan: sklearn-comparator-migration for applications/plot_*.py

## Context

The sklearn gallery uses `SklearnXorqComparator` to standardise the sklearn-vs-xorq comparison pattern across scripts. Four scripts in `applications/` have not yet been migrated. The goal is to bring them in line with the pattern where possible, expose module-level `xorq_*` prediction expressions so `xorq build` works, and fix the broken `xorq_metrics` build in `plot_time_series_lagged_features.py`.

## Verdict per script

| Script | Action | Reason |
|---|---|---|
| `plot_topics_extraction_with_nmf_lda.py` | **Skip** | Unsupervised (NMF/LDA) — no target column, no `.predict()`, output is word-weight components |
| `plot_cyclical_feature_engineering.py` | **Skip** | Uses `TimeSeriesSplit` cross-validation — fundamentally incompatible with single fit/predict model. `xorq_spline_ridge_cv`/`xorq_hgbr_cv` are score arrays, not `Expr` instances |
| `plot_tomography_l1_reconstruction.py` | **Migrate** | Ridge + Lasso, `split_data_nop`, reconstruction error read from `result["fitted"].coef_`, no overrides needed |
| `plot_time_series_lagged_features.py` | **Migrate** | Single HGBR pipeline, standard MAE/MSE metrics; move lag computation from ibis to pandas in `load_data()` |

---

## Script 1: plot_tomography_l1_reconstruction.py

**Current structure:** `sklearn_way()` / `xorq_way()` functions called at module level. No module-level `xorq_*` names → `xorq build` impossible today.

**Changes:**

1. Delete `sklearn_way()` and `xorq_way()` functions (and their module-level calls).

2. Rename / restructure `load_data()` to return a flat DataFrame:
   - Rows = projection rays
   - Feature columns = pixel weights per ray (px_0 … px_16383, 16 384 columns)
   - Target column = noisy projection measurement

3. Add `split_data = split_data_nop` import; use `df.copy()` variant (already fixed in `split_data_nop`).

4. `metrics_names_funcs = ()` — reconstruction error is not a per-sample `(y_true, y_pred)` metric; it requires `model.coef_`. Compute it in `compare_results_fn` instead.

5. `compare_results_fn(comparator)`: reads `result["fitted"].coef_` (Ridge/Lasso instance returned by default `make_sklearn_result`) to print `||image - coef_||` for sklearn and xorq.

6. `plot_results_fn(comparator)`: builds 3-panel figure (original | sklearn reconstruction | xorq reconstruction) using `result["fitted"].coef_.reshape(IMAGE_SHAPE)` for each side.

7. Module-level setup: `comparator`, then expose:
   ```python
   xorq_ridge_preds, xorq_lasso_preds = (
       comparator.deferred_xorq_results[name]["preds"] for name in methods
   )
   ```

8. `main()`: `comparator.result_comparison` then `save_fig(...)`.

**Overrides needed:** None — `result["fitted"]` is the Ridge/Lasso estimator with `.coef_` by default.

**Files to modify:** `applications/plot_tomography_l1_reconstruction.py`

---

## Script 2: plot_time_series_lagged_features.py

**Current structure:** Module-level ibis expression pipeline (`con.register`, `data_with_lags`, `train_data`, `test_data`, `fitted`, `xorq_preds`, `xorq_metrics`, `xorq_plot`). `xorq_metrics` build fails with `AttributeError: 'property' object has no attribute 'partition'`. `xorq_preds` and `xorq_plot` build fine.

**Key change:** Move lag computation from ibis (`.lag().over(order_by=ROW_IDX)`) to pandas (`.shift()`) inside `load_data()`. This is semantically equivalent — data is sorted by time index before shifting.

**Changes:**

1. Rewrite `load_data()` to:
   - Fetch OpenML Bike Sharing Demand
   - Engineer calendar features (hour, weekday, month) and weather features
   - Normalise target (`count` / `count.max()`)
   - Sort by `ROW_IDX`
   - Compute 5 lags via pandas `.shift(n)` (n = 1, 2, 3, 24, 168)
   - Drop NaN rows from lagging
   - Return flat DataFrame (drops `ROW_IDX`)

2. Remove all module-level ibis setup (`con`, `data`, `data_with_lags`, `train_data`, `test_data`, `fitted`, old `xorq_*` names).

3. `split_data = partial(train_test_split, test_size=0.333, shuffle=False)` — preserves temporal order.

4. `FEATURE_COLS`: calendar + weather + lag columns (14 total).

5. `metrics_names_funcs = (("mae", mean_absolute_error), ("mse", mean_squared_error))` — use `mse` (not a custom `rmse`) since `deferred_sklearn_metric` only accepts known sklearn scorer functions. Compute RMSE as `sqrt(mse)` in `compare_results_fn`.

6. Single pipeline: `HistGradientBoostingRegressor(max_iter=200, max_depth=8, learning_rate=0.1, random_state=42)`.

7. `compare_results_fn(comparator)`: print sklearn vs xorq MAE and RMSE (computed as `sqrt(mse)`).

8. `plot_results_fn(comparator)`: build time-series line plot using `xorq_results` (already executed DataFrames) + `fig_to_image` for side-by-side. No `deferred_matplotlib_plot` needed since results are already materialised.

9. Module-level setup: `comparator`, then expose:
   ```python
   (xorq_hgbr_preds,) = (
       comparator.deferred_xorq_results[name]["preds"] for name in methods
   )
   ```

10. `main()`: `comparator.result_comparison` then `save_fig(...)`.

**Overrides needed:** None.

**Files to modify:** `applications/plot_time_series_lagged_features.py`

---

## Verification

After migration, run:

```bash
python scripts/build_all_exprs.py
```

Expected: `xorq_ridge_preds`, `xorq_lasso_preds`, `xorq_hgbr_preds` appear in the OK list. `xorq_metrics` / `xorq_plot` / `xorq_spline_ridge_cv` / `xorq_hgbr_cv` disappear (removed or still skipped).

Also run each script directly to confirm `main()` executes without error and figures are saved:

```bash
python -c "import applications.plot_tomography_l1_reconstruction as m; m.main()"
python -c "import applications.plot_time_series_lagged_features as m; m.main()"
```

## Implementation notes

- `deferred_sklearn_metric` only accepts known sklearn scorer functions (`mean_absolute_error`, `mean_squared_error`, etc.). Custom callables like a hand-rolled `root_mean_squared_error` raise `ValueError`. Use `mse` as the metric key and compute `sqrt(mse)` in compare/plot callbacks.
- `plot_results_fn` can directly use `comparator.xorq_results[name]["preds"]` (already-executed pandas DataFrame) instead of re-running `deferred_matplotlib_plot`, avoiding a redundant second execution.
