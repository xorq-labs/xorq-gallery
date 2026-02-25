import os
import pathlib
import runpy

import click


scripts = tuple(
    p
    for p in sorted(pathlib.Path(__file__).parent.glob("sklearn/*/plot_*.py"))
    if p.name != "__init__.py"
)


def run_script(script, run_name="__main__"):
    print(f"Running {script.name}")
    return runpy.run_path(str(script), run_name=run_name)


@click.group()
def cli():
    """xorq gallery: list and run example scripts."""


@cli.command("list")
def list_scripts():
    """List available scripts."""
    click.echo("\n".join(script.stem for script in scripts))


def _complete_script_name(ctx, param, incomplete):
    return [
        click.shell_completion.CompletionItem(s.stem)
        for s in scripts
        if s.stem.startswith(incomplete)
    ]


@cli.command("run")
@click.argument("script_name", shell_complete=_complete_script_name)
def run_one(script_name):
    """Run a single script by name."""
    script = next((s for s in scripts if s.stem == script_name), None)
    match script:
        case None:
            raise click.BadParameter(
                f"no such script {script_name!r}", param_hint="script_name"
            )
        case _:
            run_script(script)


@cli.command("run-all")
def run_all_scripts():
    """Run all scripts."""
    for script in scripts:
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
