import pytest
from click.testing import CliRunner

from xorq_gallery.cli import cli
from xorq_gallery.sklearn import scripts


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
    [
        pytest.param(
            s,
            marks=pytest.mark.xfail(
                reason="SklearnXorqComparator does not accept sklearn_pipeline kwarg"
            ),
        )
        if s.stem == "plot_quantile_regression"
        else s
        for s in scripts
    ],
    ids=[s.stem for s in scripts],
)
def test_list_exprs(runner, benchmark, script):
    result = benchmark(
        runner.invoke, cli, ["list-exprs", script.stem, "-g", script.parent.name]
    )
    assert result.exit_code == 0
