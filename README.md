# xorq-gallery

A collection of scikit-learn examples ported to [xorq](https://github.com/xorq-labs/xorq) deferred ML pipelines.

Scikit-learn's [example gallery](https://scikit-learn.org/stable/auto_examples/index.html) is one of the best ways to learn practical ML. This project takes 34 of those examples, same models, same datasets, often the same code, and runs each pipeline two ways: eagerly with sklearn and deferred with xorq. Every example asserts numerical equivalence between the two and produces a side-by-side comparison plot.

## Quickstart

Run any example directly with `uv`:

```bash
uv tool run --isolated --python 3.12 --with git+ssh://git@github.com/xorq-labs/xorq-gallery xorq-gallery run plot_topics_extraction_with_nmf_lda
```

Or run locally:

```bash
xorq-gallery run plot_cyclical_feature_engineering
```

Run all examples in a group:

```bash
xorq-gallery run-all --group svm
```

## Examples

Each example lives under `src/xorq_gallery/sklearn/<category>/` and mirrors the structure of sklearn's own gallery. 34 examples across 16 categories:

### [applications](src/xorq_gallery/sklearn/applications/) (4)

- `plot_cyclical_feature_engineering` |Periodic spline features + HGBR on Bike Sharing Demand
- `plot_time_series_lagged_features` |Lagged features via pandas shift vs ibis `.lag()`
- `plot_tomography_l1_reconstruction` |Ridge vs Lasso compressive sensing on a synthetic image
- `plot_topics_extraction_with_nmf_lda` |NMF and LDA topic extraction on 20 Newsgroups

### [calibration](src/xorq_gallery/sklearn/calibration/) (1)

- `plot_compare_calibration` |Calibration curves for classifiers on a synthetic dataset

### [classification](src/xorq_gallery/sklearn/classification/) (3)

- `plot_classification_probability` |Probability estimates across classifiers on Iris
- `plot_classifier_comparison` |Side-by-side decision boundaries for 9 classifiers
- `plot_lda_qda` |LDA vs QDA decision boundaries with covariance ellipsoids

### [cluster](src/xorq_gallery/sklearn/cluster/) (3)

- `plot_affinity_propagation` |Affinity propagation clustering on synthetic data
- `plot_kmeans_digits` |K-means on handwritten digits (multiple initializations)
- `plot_kmeans_silhouette_analysis` |Silhouette analysis for choosing cluster count

### [compose](src/xorq_gallery/sklearn/compose/) (2)

- `plot_column_transformer_mixed_types` |ColumnTransformer on mixed numeric/categorical features
- `plot_feature_union` |FeatureUnion combining PCA and SelectKBest

### [decomposition](src/xorq_gallery/sklearn/decomposition/) (1)

- `plot_faces_decomposition` |PCA, NMF, ICA, and more on Olivetti faces

### [ensemble](src/xorq_gallery/sklearn/ensemble/) (3)

- `plot_gradient_boosting_categorical` |HGBR with native categorical feature support
- `plot_stack_predictors` |Stacking regressor combining spline + gradient boosting
- `plot_voting_regressor` |Voting regressor averaging GBR, RF, and linear models

### [feature_selection](src/xorq_gallery/sklearn/feature_selection/) (2)

- `plot_rfe_digits` |Recursive feature elimination on digit pixel features
- `plot_select_from_model_diabetes` |Feature importance thresholding with Lasso

### [linear_model](src/xorq_gallery/sklearn/linear_model/) (3)

- `plot_lasso_and_elasticnet` |Lasso vs ElasticNet on sparse signals
- `plot_logistic_multinomial` |Multinomial logistic regression on Iris
- `plot_quantile_regression` |Quantile regression vs OLS

### [model_selection](src/xorq_gallery/sklearn/model_selection/) (3)

- `plot_confusion_matrix` |Confusion matrix visualization for SVC on digits
- `plot_cv_indices` |Cross-validation fold visualization
- `plot_roc` |ROC curves and AUC for multiclass classification

### [neighbors](src/xorq_gallery/sklearn/neighbors/) (1)

- `plot_nca_classification` |KNN with and without Neighborhood Components Analysis

### [neural_networks](src/xorq_gallery/sklearn/neural_networks/) (1)

- `plot_mlp_alpha` |MLP regularization (alpha sweep) across 3 datasets

### [preprocessing](src/xorq_gallery/sklearn/preprocessing/) (3)

- `plot_all_scaling` |Comparing StandardScaler, MinMaxScaler, and friends
- `plot_discretization_strategies` |KBinsDiscretizer with uniform/quantile/kmeans strategies
- `plot_target_encoder` |TargetEncoder vs OrdinalEncoder on categorical features

### [svm](src/xorq_gallery/sklearn/svm/) (2)

- `plot_svm_kernels` |SVM decision boundaries with different kernels
- `plot_svm_regression` |SVR with different kernels on synthetic data

### [text](src/xorq_gallery/sklearn/text/) (1)

- `plot_document_classification_20newsgroups` |Text classification with TF-IDF + SGD/Ridge

### [tree](src/xorq_gallery/sklearn/tree/) (1)

- `plot_tree_regression` |Decision tree regression with max_depth tuning

## Build Catalog

Every deferred xorq expression produces a reproducible build artifact via `xo.build_expr`. These are stored in `builds/` and tracked in a git catalog submodule at `.xorq/git-catalogs/xorq-gallery-sklearn`.

Three caches form a chain of truth:

| Cache | File | Maps |
|-------|------|------|
| **exprs** | `src/xorq_gallery/data/exprs.json` | script name → expr names |
| **build paths** | `src/xorq_gallery/data/build_paths.json` | (script, expr) → build hash |
| **catalog** | `.xorq/git-catalogs/xorq-gallery-sklearn` | build hash → artifact, with aliases |

Catalog aliases follow the convention `{script_stem}-{expr_name}`, e.g. `plot_lasso_and_elasticnet-xorq_lasso_preds`.

### Updating

The three caches must be updated in order — each step reads from the previous:

```bash
xorq-gallery update-exprs          # scan scripts → data/exprs/{script}.json
xorq-gallery update-build-paths    # build each expr → data/build_paths/{script}.json (~7 min, -j for parallel)
xorq-gallery update-catalog        # diff build_paths vs catalog → add/remove entries and aliases
```

Use `--dry-run` with `update-catalog` to preview changes without applying them.

Equivalent Python API:

```python
from xorq_gallery.sklearn.utils import (
    update_exprs_json_cache,
    update_build_paths_json_cache,
    update_catalog,
)

update_exprs_json_cache()
update_build_paths_json_cache()
update_catalog()                  # pass dry_run=True to preview
```

### Validation

Five tests verify the chain end-to-end:

| Step | Test | Checks | Speed |
|------|------|--------|-------|
| 1 | `test_load_exprs_json_cache_matches_get_exprs_dict` | exprs.json matches live scripts | fast |
| 2 | `test_load_build_paths_json_cache_keys_match_exprs_cache` | build_paths.json covers every expr in exprs.json | fast |
| 3 | `test_build_paths_json_cache_hashes_are_current` | build hashes match what `xo.build_expr` produces now | slow (~7min) |
| 4 | `test_build_paths_json_cache_dirs_exist` | every build hash has a directory in `builds/` | fast |
| 5 | `test_catalog_matches_build_paths` | catalog entries and aliases match build_paths.json | fast |

```bash
# Run fast validation (steps 1, 2, 4, 5)
uv run pytest tests/test_utils.py -v -m "not slow2"

# Run full validation including rebuild check (all steps)
uv run pytest tests/test_utils.py -v
```

## Development

```bash
nix develop
```

The `dev/` directory is added to `PATH` via `.envrc`, so scripts can be
called by name after `direnv allow`:

```bash
rebuild-catalog-git.sh          # fresh empty catalog (plain git)
rebuild-catalog-annex.sh --gcs  # fresh empty catalog (git-annex + S3)
switch-catalog-branch.sh main   # switch to a branch with a different remote
```

Run all examples:

```bash
xorq-gallery run-all
```

Run tests:

```bash
nix develop -c pytest tests/test_examples.py -v
```

## Output

Examples write comparison plots to `imgs/`. This directory is created automatically but is not tracked by git.


## git-annex catalog

The `feat/catalog/git-annex` branch backs the catalog submodule with
[git-annex](https://git-annex.branchable.com/) and an S3 remote. Entry
`.zip` artifacts are stored in S3 instead of committed to git, keeping the
repo small. Read-only credentials are embedded in the annex repo so no
extra setup is needed.

### Cloning

```bash
git clone --branch=feat/catalog/git-annex --recurse-submodules https://github.com/xorq-labs/xorq-gallery
```

After cloning, entry files are broken symlinks. Content is fetched lazily
the first time a `CatalogEntry` is accessed (via `entry.expr` or
`entry.fetch()`). The `Catalog` class auto-detects the annex branch,
runs `annex init`, and enables the S3 remote.

### Switching branches

Plain `git checkout` does not sync submodule URLs. Use the helper script
when switching between branches that use different catalog remotes (e.g.
`main` uses plain git, `feat/catalog/git-annex` uses annex+S3):

```bash
./dev/switch-catalog-branch.sh <branch>
```

Run with no arguments to repair the submodule on the current branch:

```bash
./dev/switch-catalog-branch.sh
```
