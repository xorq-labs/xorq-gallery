import json
import os
import sys
from pathlib import Path

import pytest

from xorq_gallery.cli import PINNED_PYTHON
from xorq_gallery.sklearn import get_exprs_for_script, scripts
from xorq_gallery.sklearn.utils import (
    _current_catalog_state,
    _desired_catalog_state,
    _get_catalog,
    get_build_paths_dict,
    load_build_paths_json_cache,
    load_exprs_json_cache,
)


def test_python_version_matches_pinned():
    """Step 0: fail fast if Python minor version doesn't match PINNED_PYTHON."""
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert current == PINNED_PYTHON, (
        f"Running Python {current}, but build hashes require {PINNED_PYTHON}. "
        f"Update the CI matrix or PINNED_PYTHON in cli.py."
    )


_XFAIL_SCRIPTS = {
    "plot_document_classification_20newsgroups": "20 Newsgroups dataset download unreliable in CI",
}


@pytest.mark.parametrize(
    "script",
    [
        pytest.param(
            s,
            id=s.stem,
            marks=[pytest.mark.xfail(reason=_XFAIL_SCRIPTS[s.stem])]
            if s.stem in _XFAIL_SCRIPTS
            else [],
        )
        for s in scripts
    ],
)
def test_exprs_snapshot(script, snapshot):
    """Step 1: exprs discovered from each script match snapshot."""
    exprs = get_exprs_for_script(script)
    snapshot.assert_match(
        json.dumps(list(exprs.keys()), indent=2),
        f"exprs/{script.name}.json",
    )


def test_load_build_paths_json_cache_keys_match_exprs_cache():
    """Step 2: build_paths.json covers every expr in exprs.json."""
    exprs_cache = load_exprs_json_cache()
    build_cache = load_build_paths_json_cache()
    for script_name, expr_names in exprs_cache.items():
        assert script_name in build_cache, f"missing script: {script_name}"
        for expr_name in expr_names:
            assert expr_name in build_cache[script_name], (
                f"missing expr: {script_name}:{expr_name}"
            )


def _changed_script_names():
    """Return script names to check, or None for all.

    When CHANGED_SCRIPTS is set (comma-separated basenames, with or without .py),
    only those scripts are rebuilt. Unset or empty means check all.
    """
    raw = os.environ.get("CHANGED_SCRIPTS", "").strip()
    if not raw:
        return None
    return tuple(s if s.endswith(".py") else f"{s}.py" for s in raw.split(","))


_NONDETERMINISTIC_SCRIPTS = frozenset(
    {
        "plot_faces_decomposition.py",  # NMF, ICA, DictionaryLearning converge differently across platforms
    }
)


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

_IMPLICIT_PARALLEL_SCRIPTS = frozenset(
    {
        "plot_cyclical_feature_engineering",
        "plot_target_encoder",
        "plot_gradient_boosting_categorical",
        "plot_faces_decomposition",
        "plot_tomography_l1_reconstruction",
        "plot_kmeans_digits",
    }
)


def _marks_for_hash_check(script_name):
    stem = script_name.removesuffix(".py")
    marks = []
    if stem in _XFAIL_SCRIPTS:
        marks.append(pytest.mark.xfail(reason=_XFAIL_SCRIPTS[stem]))
    if script_name in _NONDETERMINISTIC_SCRIPTS:
        marks.append(pytest.mark.xfail(reason="non-deterministic across platforms"))
    if stem in _SLOW1_SCRIPTS:
        marks.append(pytest.mark.slow1)
    if stem in _SLOW2_SCRIPTS:
        marks.append(pytest.mark.slow2)
    if stem in _IMPLICIT_PARALLEL_SCRIPTS:
        marks.append(pytest.mark.implicit_parallel)
    return marks


def _hash_check_scripts():
    """Build parametrize list for per-script hash checks."""
    cached = load_build_paths_json_cache()
    changed = _changed_script_names()
    scripts = changed if changed else tuple(cached.keys())
    return [
        pytest.param(s, id=s.removesuffix(".py"), marks=_marks_for_hash_check(s))
        for s in scripts
    ]


@pytest.mark.parametrize("script_name", _hash_check_scripts())
def test_build_hashes_snapshot(script_name, snapshot):
    """Step 3: build hashes for each script match snapshot."""
    rebuilt = get_build_paths_dict(script_names=(script_name,))
    snapshot.assert_match(
        json.dumps(rebuilt.get(script_name, {}), sort_keys=True, indent=2),
        f"build_hashes/{script_name}.json",
    )


def test_build_paths_json_cache_entries_exist_in_catalog():
    """Step 4: every build hash in build_paths.json exists in the catalog."""
    build_cache = load_build_paths_json_cache()
    catalog = _get_catalog()
    catalog_entries = frozenset(catalog.list())
    missing = tuple(
        (script_name, expr_name, build_path)
        for script_name, exprs in build_cache.items()
        for expr_name, build_path in exprs.items()
        if Path(build_path).name not in catalog_entries
    )
    assert not missing, "Missing catalog entries:\n" + "\n".join(
        f"  {sn}:{en} -> {bp}" for sn, en, bp in missing
    )


def test_catalog_matches_build_paths():
    """Step 5: catalog entries/aliases match build_paths.json."""
    build_cache = load_build_paths_json_cache()
    catalog = _get_catalog()

    desired_entries, desired_alias_to_entry = _desired_catalog_state(build_cache)
    current_entries, current_alias_to_entry = _current_catalog_state(catalog)

    assert current_entries == desired_entries, (
        f"extra={sorted(current_entries - desired_entries)}, "
        f"missing={sorted(desired_entries - current_entries)}"
    )
    assert set(current_alias_to_entry) == set(desired_alias_to_entry), (
        f"alias mismatches: {set(current_alias_to_entry) ^ set(desired_alias_to_entry)}"
    )
