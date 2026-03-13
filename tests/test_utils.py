import os
import sys
from pathlib import Path

import pytest

from xorq_gallery.cli import PINNED_PYTHON
from xorq_gallery.sklearn.utils import (
    _current_catalog_state,
    _desired_catalog_state,
    _get_catalog,
    get_build_paths_dict,
    get_exprs_dict,
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


def test_load_exprs_json_cache_matches_get_exprs_dict():
    """Step 1: exprs.json matches live scripts."""
    cached = load_exprs_json_cache()
    calculated = get_exprs_dict()
    left, right = (
        set((k, tuple(v)) for k, v in el.items()) for el in (cached, calculated)
    )
    assert left == right, (left.difference(right), right.difference(left))


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


@pytest.mark.slow2
def test_build_paths_json_cache_hashes_are_current():
    """Step 3: build_paths.json hashes match what xo.build_expr produces."""
    cached = load_build_paths_json_cache()
    script_filter = _changed_script_names()
    rebuilt = get_build_paths_dict(script_names=script_filter)
    scripts_to_check = rebuilt.keys() if script_filter else cached.keys()
    mismatches = tuple(
        (script_name, expr_name, cached[script_name][expr_name], rebuilt_path)
        for script_name in scripts_to_check
        if script_name in cached
        for expr_name, cached_path in cached[script_name].items()
        if (rebuilt_path := rebuilt.get(script_name, {}).get(expr_name)) != cached_path
    )
    assert not mismatches, "Stale hashes in build_paths.json:\n" + "\n".join(
        f"  {sn}:{en}: cached={cp}, rebuilt={rp}" for sn, en, cp, rp in mismatches
    )


def test_build_paths_json_cache_dirs_exist():
    """Step 4: every build hash in build_paths.json exists on disk."""
    build_cache = load_build_paths_json_cache()
    builds_dir = Path(__file__).resolve().parent.parent / "builds"
    missing = tuple(
        (script_name, expr_name, build_path)
        for script_name, exprs in build_cache.items()
        for expr_name, build_path in exprs.items()
        if not (builds_dir / Path(build_path).name).is_dir()
    )
    assert not missing, "Missing build directories:\n" + "\n".join(
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
