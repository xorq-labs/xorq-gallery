import pathlib
import runpy

import pytest
from pytest import param


sklearn_dir = (
    pathlib.Path(__file__).parents[1]
    / "src"
    / "xorq_gallery"
    / "sklearn"
)

# Collect all example scripts from all categories (applications, calibration, etc.)
scripts = []
for category_dir in sorted(sklearn_dir.iterdir()):
    if category_dir.is_dir() and not category_dir.name.startswith("_"):
        for script in sorted(category_dir.glob("plot_*.py")):
            scripts.append((category_dir.name, script))


@pytest.mark.parametrize(
    "category,script",
    [param(cat, script, id=f"{cat}/{script.stem}") for cat, script in scripts],
)
def test_script_execution(category, script):
    dct = runpy.run_path(str(script), run_name="__pytest_main__")
    assert dct.get("pytest_examples_passed")
