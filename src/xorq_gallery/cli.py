import os
import pathlib
import pdb
import runpy
import subprocess
import sys
import traceback

import click

from xorq_gallery.sklearn import (
    get_scripts_for_group,
    group_paths,
    scripts,
)
from xorq_gallery.sklearn.utils import (
    load_exprs_json_cache,
    update_build_paths_json_cache,
    update_exprs_json_cache,
)


PINNED_PYTHON = "3.13"


def _reexec_if_needed(ctx):
    """Re-exec the current command under the pinned Python via ``uv run`` if needed."""
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current == PINNED_PYTHON:
        return False
    args = ["uv", "run", "--python", PINNED_PYTHON, "xorq-gallery"]
    # reconstruct full command path from click context
    parts = []
    c = ctx
    while c and c.info_name != ctx.find_root().info_name:
        parts.append(c.info_name)
        c = c.parent
    args.extend(reversed(parts))
    args.extend(ctx.params.get("script_names", ()))
    click.echo(f"Re-running under Python {PINNED_PYTHON}: {' '.join(args)}")
    sys.exit(subprocess.run(args).returncode)


def _scripts_for_group(group):
    try:
        return get_scripts_for_group(group)
    except ValueError:
        raise click.BadParameter(f"no such group {group!r}", param_hint="group")


def run_script(script, run_name="__main__"):
    print(f"Running {script.name}")
    dct = runpy.run_path(str(script), run_name=run_name)
    dct["main"]()
    return dct


def _complete_group(ctx, param, incomplete):
    return [
        click.shell_completion.CompletionItem(g.name)
        for g in group_paths
        if g.name.startswith(incomplete)
    ]


def _complete_script_name(ctx, param, incomplete):
    group = ctx.params.get("group")
    pool = _scripts_for_group(group) if group else scripts
    return [
        click.shell_completion.CompletionItem(s.stem)
        for s in pool
        if s.stem.startswith(incomplete)
    ]


class PdbCommand(click.Command):
    def invoke(self, ctx):
        if ctx.parent and ctx.parent.params.get("pdb_runcall"):
            return pdb.runcall(ctx.invoke, self.callback, **ctx.params)
        return super().invoke(ctx)


class PdbGroup(click.Group):
    command_class = PdbCommand

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except (click.ClickException, click.exceptions.Exit, SystemExit):
            raise
        except Exception as e:
            if ctx.params.get("use_pdb"):
                traceback.print_exception(e)
                pdb.post_mortem(e.__traceback__)
            else:
                traceback.print_exc()
            sys.exit(1)


@click.group(cls=PdbGroup)
@click.option("--pdb", "use_pdb", is_flag=True, help="Drop into pdb on failure")
@click.option(
    "--pdb-runcall", "pdb_runcall", is_flag=True, help="Invoke with pdb.runcall"
)
def cli(use_pdb, pdb_runcall):
    """xorq gallery: list and run example scripts."""


@cli.command("list-groups")
def list_groups():
    """List available script groups."""
    click.echo("\n".join(g.name for g in group_paths))


@cli.command("list")
@click.option(
    "-g",
    "--group",
    default=None,
    shell_complete=_complete_group,
    help="Filter by group.",
)
def list_scripts(group):
    """List available scripts."""
    pool = _scripts_for_group(group) if group else scripts
    click.echo("\n".join(s.stem for s in pool))


@cli.command("list-exprs")
@click.argument("script_name", shell_complete=_complete_script_name)
@click.option(
    "-g",
    "--group",
    default=None,
    shell_complete=_complete_group,
    help="Restrict search to a group.",
)
def list_exprs(script_name, group):
    """Run a single script by name."""
    pool = _scripts_for_group(group) if group else scripts
    script = next((s for s in pool if s.stem == script_name), None)
    match script:
        case None:
            raise click.BadParameter(
                f"no such script {script_name!r}", param_hint="script_name"
            )
        case _:
            cache = load_exprs_json_cache()
            exprs = cache.get(script.name, [])
            click.echo("\n".join(exprs))


@cli.command("run")
@click.argument("script_name", shell_complete=_complete_script_name)
@click.option(
    "-g",
    "--group",
    default=None,
    shell_complete=_complete_group,
    help="Restrict search to a group.",
)
def run_one(script_name, group):
    """Run a single script by name."""
    pool = _scripts_for_group(group) if group else scripts
    script = next((s for s in pool if s.stem == script_name), None)
    match script:
        case None:
            raise click.BadParameter(
                f"no such script {script_name!r}", param_hint="script_name"
            )
        case _:
            run_script(script)


@cli.command("run-all")
@click.option(
    "-g",
    "--group",
    default=None,
    shell_complete=_complete_group,
    help="Restrict to a group.",
)
def run_all_scripts(group):
    """Run all scripts."""
    pool = _scripts_for_group(group) if group else scripts
    for script in pool:
        run_script(script)


_COMPLETION_INSTALL_PATHS = {
    "bash": pathlib.Path(
        "~/.local/share/bash-completion/completions/xorq-gallery"
    ).expanduser(),
    "zsh": pathlib.Path("~/.zfunc/_xorq-gallery").expanduser(),
    "fish": pathlib.Path("~/.config/fish/completions/xorq-gallery.fish").expanduser(),
}


def _get_completion_source(shell):
    from click.shell_completion import get_completion_class

    comp_cls = get_completion_class(shell)
    comp = comp_cls(cli, {}, "xorq-gallery", "_XORQ_GALLERY_COMPLETE")
    return comp.source()


def _detect_shell():
    shell_bin = pathlib.Path(os.environ.get("SHELL", "")).name
    if shell_bin not in _COMPLETION_INSTALL_PATHS:
        raise click.UsageError(
            f"Cannot detect shell from $SHELL={os.environ.get('SHELL')!r}. "
            "Pass the shell name explicitly: bash, zsh, or fish."
        )
    return shell_bin


@cli.command("completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]), required=False)
def completion(shell):
    """Output shell completion script.

    SHELL defaults to the value of $SHELL if not provided.

    \b
    Add to your shell config:
      bash:  eval "$(xorq-gallery completion bash)"
      zsh:   eval "$(xorq-gallery completion zsh)"
      fish:  xorq-gallery completion fish | source
    """
    if shell is None:
        shell = _detect_shell()
    click.echo(_get_completion_source(shell), nl=False)


@cli.command("install-completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]), required=False)
def install_completion(shell):
    """Install shell completion to the standard location.

    SHELL defaults to the value of $SHELL if not provided.

    \b
    Install paths:
      bash:  ~/.local/share/bash-completion/completions/xorq-gallery
      zsh:   ~/.zfunc/_xorq-gallery  (requires ~/.zfunc in fpath)
      fish:  ~/.config/fish/completions/xorq-gallery.fish
    """
    if shell is None:
        shell = _detect_shell()
    install_path = _COMPLETION_INSTALL_PATHS[shell]
    install_path.parent.mkdir(parents=True, exist_ok=True)
    install_path.write_text(_get_completion_source(shell))
    click.echo(f"Installed {shell} completion to {install_path}")
    click.echo(f"Restart your shell or run: source {install_path}")


@cli.command("update-exprs")
@click.pass_context
def update_exprs(ctx):
    """Rebuild exprs.json cache from current scripts."""
    _reexec_if_needed(ctx)
    with click.progressbar(
        length=len(scripts), label="Scanning scripts", item_show_func=str
    ) as bar:

        def _on_script(name):
            bar.current_item = name
            bar.update(1)

        path = update_exprs_json_cache(on_script=_on_script)
    click.echo(f"Updated {path}")


@cli.command("update-build-paths")
@click.argument("script_names", nargs=-1, shell_complete=_complete_script_name)
@click.option(
    "-j",
    "--workers",
    type=int,
    default=None,
    help="Max parallel workers (default: cpu_count). Use -j1 for sequential (required for --pdb).",
)
@click.pass_context
def update_build_paths(ctx, script_names, workers):
    """Rebuild build_paths.json cache from current exprs.

    Optionally pass one or more SCRIPT_NAMES to update only those scripts.
    """
    _reexec_if_needed(ctx)
    from xorq_gallery.sklearn.utils import load_exprs_json_cache as _load_exprs

    if workers is None:
        workers = os.cpu_count()

    filter_names = (
        tuple(s if s.endswith(".py") else f"{s}.py" for s in script_names) or None
    )
    cache = _load_exprs()
    total = sum(1 for sn in cache if filter_names is None or sn in filter_names)
    with click.progressbar(
        length=total, label="Building exprs", item_show_func=str
    ) as bar:

        def _on_script(name):
            bar.current_item = name
            bar.update(1)

        path = update_build_paths_json_cache(
            script_names=filter_names, on_script=_on_script, max_workers=workers
        )
    if filter_names:
        click.echo(f"Updated {path} for {', '.join(filter_names)}")
    else:
        click.echo(f"Updated {path}")


@cli.command("update-catalog")
@click.option("--dry-run", is_flag=True, help="Show diff without applying.")
def update_catalog_cmd(dry_run):
    """Sync the git catalog submodule with build_paths.json."""
    from git import Repo

    from xorq_gallery.sklearn.utils import (
        _get_catalog,
        compute_catalog_diff,
    )
    from xorq_gallery.sklearn.utils import (
        load_build_paths_json_cache as _load_build_paths,
    )

    catalog = _get_catalog()
    repo_root = pathlib.Path(
        Repo(pathlib.Path.cwd(), search_parent_directories=True).working_dir
    )
    builds_dir = repo_root / "builds"
    build_cache = _load_build_paths()

    diff = compute_catalog_diff(catalog, build_cache)
    if diff.is_empty:
        click.echo("Catalog is up to date.")
        return
    if diff.entries_to_add:
        click.echo(f"Entries to add: {len(diff.entries_to_add)}")
    if diff.entries_to_remove:
        click.echo(f"Entries to remove: {len(diff.entries_to_remove)}")
    if diff.aliases_to_add:
        click.echo(f"Aliases to add: {len(diff.aliases_to_add)}")
    if diff.aliases_to_remove:
        click.echo(f"Aliases to remove: {len(diff.aliases_to_remove)}")
    if dry_run:
        click.echo("(dry run — no changes applied)")
        return

    from xorq.catalog.catalog import CatalogAlias

    if diff.aliases_to_remove:
        with click.progressbar(
            diff.aliases_to_remove,
            label="Removing aliases",
            item_show_func=str,
        ) as bar:
            for alias in bar:
                CatalogAlias.from_name(alias, catalog).remove()

    if diff.entries_to_remove:
        with click.progressbar(
            diff.entries_to_remove,
            label="Removing entries",
            item_show_func=str,
        ) as bar:
            for entry_name in bar:
                catalog.remove(entry_name, sync=False)

    if diff.entries_to_add:
        with click.progressbar(
            diff.entries_to_add,
            label="Adding entries",
            item_show_func=lambda x: x[0][:12] if x else "",
        ) as bar:
            for entry_hash, aliases in bar:
                build_dir = builds_dir / entry_hash
                assert build_dir.is_dir(), f"Build directory not found: {build_dir}"
                catalog.add(build_dir, sync=False, aliases=aliases)

    if diff.aliases_to_add:
        with click.progressbar(
            diff.aliases_to_add,
            label="Adding aliases",
            item_show_func=lambda x: x[0] if x else "",
        ) as bar:
            for alias, entry_hash in bar:
                catalog.add_alias(entry_hash, alias, sync=False)

    catalog.assert_consistency()


@cli.command("update-snapshots")
@click.argument("script_names", nargs=-1, shell_complete=_complete_script_name)
@click.pass_context
def update_snapshots(ctx, script_names):
    """Update pytest snapshots for exprs and build-hashes tests.

    Optionally pass one or more SCRIPT_NAMES to update only those scripts.
    """
    _reexec_if_needed(ctx)
    from git import Repo

    repo_root = pathlib.Path(
        Repo(pathlib.Path.cwd(), search_parent_directories=True).working_dir
    )
    test_file = str(repo_root / "tests" / "test_utils.py")
    args = [
        "pytest",
        "--verbose",
        "--import-mode=importlib",
        "--snapshot-update",
        test_file,
    ]
    if script_names:
        k_expr = " or ".join(s.removesuffix(".py") for s in script_names)
        args.extend(["-k", k_expr])
    sys.exit(subprocess.run(args).returncode)


@cli.command("pytest-changed")
@click.option("--base", default="main", help="Base branch/ref to diff against.")
@click.argument("pytest_args", nargs=-1, type=click.UNPROCESSED)
def pytest_changed(base, pytest_args):
    """Run pytest with CHANGED_SCRIPTS set from git diff against BASE.

    Only sklearn scripts changed since BASE are hash-checked. Extra args
    are forwarded to pytest.

    \b
    Examples:
      xorq-gallery test
      xorq-gallery test --base HEAD~3
      xorq-gallery test -- -v -m slow2
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", base, "--", "src/xorq_gallery/sklearn/**/*.py"],
        capture_output=True,
        text=True,
    )
    changed = ",".join(
        sorted({pathlib.Path(f).name for f in result.stdout.strip().splitlines()})
    )
    if changed:
        click.echo(f"Changed scripts: {changed}")
    else:
        click.echo("No sklearn scripts changed; checking all.")

    env = {**os.environ, "CHANGED_SCRIPTS": changed}
    from git import Repo

    repo_root = pathlib.Path(
        Repo(pathlib.Path.cwd(), search_parent_directories=True).working_dir
    )
    has_paths = any(not a.startswith("-") for a in pytest_args)
    args = ["pytest", "--verbose", "--import-mode=importlib", *pytest_args]
    if not has_paths:
        args.append(str(repo_root / "tests"))
    sys.exit(subprocess.run(args, env=env).returncode)
