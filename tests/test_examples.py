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
    "plot_quantile_regression": "SklearnXorqComparator does not accept sklearn_pipeline kwarg",
}


@pytest.mark.parametrize(
    "category,script",
    [
        param(
            cat,
            script,
            id=f"{cat}/{script.stem}",
            marks=pytest.mark.xfail(reason=_XFAIL_SCRIPTS[script.stem]),
        )
        if script.stem in _XFAIL_SCRIPTS
        else param(cat, script, id=f"{cat}/{script.stem}")
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
