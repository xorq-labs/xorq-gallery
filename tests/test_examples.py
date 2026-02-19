import pathlib
import runpy

import pytest
from pytest import param


examples_dir = pathlib.Path(__file__).parents[1] / "src" / "xorq_gallery" / "sklearn" / "applications"
scripts = sorted(examples_dir.glob("*.py"))


@pytest.mark.parametrize(
    "script",
    [param(script, id=script.stem) for script in scripts],
)
def test_script_execution(script):
    dct = runpy.run_path(str(script), run_name="__pytest_main__")
    assert dct.get("pytest_examples_passed")
