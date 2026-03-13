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


def _marks_for(stem):
    marks = []
    if stem in _XFAIL_SCRIPTS:
        marks.append(pytest.mark.xfail(reason=_XFAIL_SCRIPTS[stem]))
    if stem in _SLOW1_SCRIPTS:
        marks.append(pytest.mark.slow1)
    if stem in _SLOW2_SCRIPTS:
        marks.append(pytest.mark.slow2)
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
