# Migrating plot_*.py files to SklearnXorqComparator

## Reference implementations
- **Done (use as models):** `cluster/plot_kmeans_digits.py`, `linear_model/plot_lasso_and_elasticnet.py`
- **Shared infrastructure:** `sklearn_lib.py` — `SklearnXorqComparator`, `split_data_nop`, `make_sklearn_result`, `make_deferred_xorq_result`, `make_xorq_result`

## Standard steps

1. Define module-level constants — `FEATURE_COLS`, `TARGET_COL`, `PRED_COL`, dataset shape assertions
2. `load_data()` returns a flat pandas DataFrame with features + target
3. `names_pipelines` — `tuple[tuple[str, SklearnPipeline], ...]`, one per model/variant
4. `metrics_names_funcs` — `tuple[tuple[str, callable], ...]` with signature `(y_true, y_pred)`
5. `split_data` — `split_data_nop` if no split needed, else curried `train_test_split`
6. `compare_results(comparator)` — prints `comparator.sklearn_results` vs `comparator.xorq_results` per metric
7. `plot_results(comparator)` — builds and returns a composite `fig`; use `fig_to_image` for both sides
8. Module-level `comparator = SklearnXorqComparator(...)` instantiation
9. Expose deferred preds at module level: `(xorq_foo_preds, ...) = (comparator.deferred_xorq_results[name]["preds"] for name in methods)`
10. `main()` — calls `comparator.result_comparison`, then `save_fig(..., comparator.plot_results())`
11. `if __name__ in ("__pytest_main__",):`

## File-specific decision table

| Situation | What to do |
|---|---|
| Metric needs more than `(y_true, y_pred)` (e.g. silhouette, AUC curve) | Compute outside comparator in `main()` using `comparator.df` and `comparator.sklearn_results` |
| Need to extract fitted model attributes (e.g. `coef_`, `ranking_`) | Define `make_sklearn_other`/`make_xorq_other`, pass curried overrides as `make_sklearn_result`/`make_deferred_xorq_result` args |
| `plot_results` needs feature columns but preds table only has `[target, pred]` | Use `xo.memtable(comparator.df)` instead of the preds table in `deferred_matplotlib_plot` |
| `plot_results` needs the full fitted pipeline (e.g. decision boundary with StandardScaler+NCA) | `result["fitted"]` is only the LAST pipeline step. Use `make_sklearn_result` override with `make_other=lambda fitted: {"full_pipeline": fitted}` to store the full pipeline; access via `result["other"]["full_pipeline"]` |
| Clustering: need `labels_`, `cluster_centers_indices_` from fitted estimator | `result["fitted"]` IS the estimator directly (last step). Access `af_fitted.labels_` etc. directly — no `.named_steps` needed |
| `plot_results` needs to predict on a different grid (e.g. smooth linspace) | Use `make_sklearn_result` override with `make_other` to store full fitted pipeline; use `comparator.deferred_xorq_results[name]["xorq_fitted"].predict(xo.memtable(plot_df), name=PRED_COL).execute()` for xorq |
| Multiple datasets × multiple models | One `comparator` per dataset, or a `dict[str, SklearnXorqComparator]` keyed by dataset name |
| Ensemble wrappers (Stacking/Voting tuple→list) | Keep existing wrapper class; pass wrapped pipeline into `names_pipelines`; add `self.n_features_in_ = X.shape[1]` in wrapper's `fit` so `check_is_fitted` passes inside `SklearnPipeline` |
| CV-based metrics (`cross_val_score`, `deferred_cross_val_score`) | Does not fit comparator pattern — leave as-is |
| No model fitting (pure visualization, e.g. CV split plots) | Skip — `SklearnXorqComparator` does not apply |
| fit_transform pipelines (scalers, discretizers) | Skip — comparator assumes fit/predict; transform-only steps don't have a `predict` |

## Migration priority

**Easy** (standard fit/predict/metric, single dataset — do these first):
- ~~`model_selection/plot_confusion_matrix.py`~~ — done
- ~~`cluster/plot_affinity_propagation.py`~~ — done
- ~~`neighbors/plot_nca_classification.py`~~ — done
- ~~`compose/plot_column_transformer_mixed_types.py`~~ — done
- ~~`classification/plot_lda_qda.py`~~ — done

**Medium** (extra model attributes, ensemble wrappers, or multi-dataset):
- `cluster/plot_kmeans_silhouette_analysis.py`
- `feature_selection/plot_rfe_digits.py`
- ~~`ensemble/plot_voting_regressor.py`~~ — done
- ~~`ensemble/plot_stack_predictors.py`~~ — done
- `classification/plot_classification_probability.py`
- `svm/plot_svm_regression.py`
- `svm/plot_svm_kernels.py`

**Hard / out of scope** (CV-based, transform-only, or no model):
- `model_selection/plot_cv_indices.py` — no model
- `compose/plot_feature_union.py` — GridSearchCV
- `preprocessing/plot_all_scaling.py` — fit_transform only
- `preprocessing/plot_discretization_strategies.py` — fit_transform only
- `preprocessing/plot_target_encoder.py` — cross_val_score
- `model_selection/plot_roc.py` — per-class AUC composition
- `neural_networks/plot_mlp_alpha.py` — large param grid, complex deferred plot
- `classification/plot_classifier_comparison.py` — 8 classifiers × 3 datasets
- `ensemble/plot_gradient_boosting_categorical.py` — cross_val_score
