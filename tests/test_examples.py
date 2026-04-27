import os
import pathlib
import runpy

# Set non-interactive matplotlib backend before any imports
import matplotlib as mpl
import pytest
from pytest import param

from xorq_gallery.sklearn import (
    get_scripts_for_group,
    group_paths,
)


mpl.use("Agg")


repo_root = pathlib.Path(__file__).parents[1]
imgs_dir = repo_root / "imgs"

# Collect all example scripts from all categories (applications, calibration, etc.)
scripts = tuple(
    (group, script)
    for group in (group.name for group in group_paths)
    for script in get_scripts_for_group(group)
)


_XFAIL_SCRIPTS = {
    "plot_topics_extraction_with_nmf_lda": "xorq#1713: do_into_backend skipped for KV-encoded transforms breaks downstream steps",
    "plot_stack_predictors": "xorq#1713: do_into_backend skipped for KV-encoded transforms breaks downstream steps",
    "plot_document_classification_20newsgroups": "20 Newsgroups dataset download unreliable in CI",
    "plot_column_transformer_mixed_types": "OpenML fetch_openml redirect loop in CI",
    "plot_time_series_lagged_features": "OpenML fetch_openml redirect loop in CI",
    "plot_cyclical_feature_engineering": "OpenML fetch_openml redirect loop in CI",
    "plot_target_encoder": "OpenML fetch_openml redirect loop in CI",
    "plot_gradient_boosting_categorical": "OpenML fetch_openml redirect loop in CI",
}

_SLOW1_SCRIPTS = frozenset(
    {
        "plot_time_series_lagged_features",
        "plot_discretization_strategies",
        "plot_all_scaling",
        "plot_compare_calibration",
        "plot_document_classification_20newsgroups",
        "plot_classifier_comparison",
        "plot_select_from_model_diabetes",
        "plot_kmeans_digits",
        "plot_kmeans_silhouette_analysis",
    }
)

_SLOW2_SCRIPTS = frozenset(
    {
        "plot_target_encoder",
        "plot_faces_decomposition",
        "plot_tomography_l1_reconstruction",
        "plot_stack_predictors",
        "plot_cyclical_feature_engineering",
        "plot_feature_union",
        "plot_topics_extraction_with_nmf_lda",
    }
)

# Scripts that use OpenMP (HistGradientBoosting, KMeans) or heavy BLAS/LAPACK
# (PCA, NMF, Ridge on large matrices) internally. These should not run under
# pytest-xdist as the implicit threading causes CPU contention.
_IMPLICIT_PARALLEL_SCRIPTS = frozenset(
    {
        "plot_cyclical_feature_engineering",  # HistGradientBoostingRegressor (OpenMP)
        "plot_target_encoder",  # HistGradientBoostingRegressor (OpenMP)
        "plot_gradient_boosting_categorical",  # 5× HistGradientBoostingRegressor (OpenMP)
        "plot_faces_decomposition",  # PCA, NMF, FastICA, SparsePCA, DictLearning (BLAS)
        "plot_tomography_l1_reconstruction",  # Ridge, Lasso on 16K×16K matrix (BLAS)
        "plot_kmeans_digits",  # KMeans (OpenMP), PCA (BLAS)
    }
)


def _marks_for(stem):
    marks = []
    if stem in _XFAIL_SCRIPTS:
        marks.append(pytest.mark.xfail(reason=_XFAIL_SCRIPTS[stem]))
    if stem in _SLOW1_SCRIPTS:
        marks.append(pytest.mark.slow1)
    if stem in _SLOW2_SCRIPTS:
        marks.append(pytest.mark.slow2)
    if stem in _IMPLICIT_PARALLEL_SCRIPTS:
        marks.append(pytest.mark.implicit_parallel)
    return marks


@pytest.mark.parametrize(
    "category,script",
    [
        param(cat, script, id=f"{cat}/{script.stem}", marks=_marks_for(script.stem))
        for cat, script in scripts
    ],
)
def test_script_execution(category, script):
    # Ensure imgs directory exists
    imgs_dir.mkdir(exist_ok=True)

    # Change to repo root so relative "imgs/" paths work
    original_cwd = pathlib.Path.cwd()
    os.chdir(repo_root)

    try:
        dct = runpy.run_path(str(script), run_name="__pytest_main__")
        assert dct.get("pytest_examples_passed")
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
