# xorq-gallery

Side-by-side examples comparing **sklearn** (eager) and **xorq** (deferred) approaches to common ML tasks. Each example shows the sklearn way and the xorq way, producing identical results.

## Quickstart

```bash
nix develop
python -m pytest -s
```

To run a single example:

```bash
python -m pytest -s -k plot_cyclical_feature_engineering
```

## Examples

| Example | Dataset | What it demonstrates |
|---------|---------|---------------------|
| `plot_cyclical_feature_engineering` | Bike Sharing Demand | Periodic spline features + HGBR, temporal split |
| `plot_time_series_lagged_features` | Bike Sharing Demand | Lagged features via pandas shift vs ibis `.lag()` |
| `plot_tomography_l1_reconstruction` | Synthetic 128x128 image | Ridge (L2) vs Lasso (L1) compressive sensing |
| `plot_topics_extraction_with_nmf_lda` | 20 Newsgroups | NMF and LDA topic extraction |

## Output

Examples write plots to `imgs/`. This directory is created automatically but is not tracked by git. If you want to ignore it, add `imgs/` to your own `.gitignore`.
