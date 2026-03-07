import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from xorq_gallery.cli import cli
from xorq_gallery.sklearn import scripts


pytestmark = pytest.mark.benchmark

_CLI = Path(sys.executable).with_name("xorq-gallery")


def _run(args):
    return subprocess.run([_CLI] + args, capture_output=True)


@pytest.fixture(scope="module")
def runner():
    return CliRunner()


def test_list_groups(runner, benchmark):
    result = benchmark(runner.invoke, cli, ["list-groups"])
    assert result.exit_code == 0


def test_list_all_scripts(runner, benchmark):
    result = benchmark(runner.invoke, cli, ["list"])
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "script",
    list(scripts),
    ids=[s.stem for s in scripts],
)
def test_list_exprs(runner, benchmark, script):
    result = benchmark(
        runner.invoke, cli, ["list-exprs", script.stem, "-g", script.parent.name]
    )
    assert result.exit_code == 0


def test_subprocess_list_groups(benchmark):
    result = benchmark(_run, ["list-groups"])
    assert result.returncode == 0


def test_subprocess_list_all_scripts(benchmark):
    result = benchmark(_run, ["list"])
    assert result.returncode == 0


@pytest.mark.parametrize(
    "script_stem,group",
    [
        ("plot_lasso_and_elasticnet", "linear_model"),
        ("plot_confusion_matrix", "model_selection"),
        ("plot_tree_regression", "tree"),
    ],
)
def test_subprocess_list_exprs(benchmark, script_stem, group):
    result = benchmark(_run, ["list-exprs", script_stem, "-g", group])
    assert result.returncode == 0
