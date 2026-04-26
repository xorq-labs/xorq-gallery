import importlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from attr import frozen
from toolz.curried import excepts as cexcepts

from xorq_gallery.sklearn import (
    get_exprs_for_script,
    scripts,
)
from xorq_gallery.utils import get_data_file


exprs_dir = get_data_file("exprs")
build_paths_dir = get_data_file("build_paths")


def _module_name_for_script(script):
    parts = script.parts
    idx = next(i for i, p in enumerate(parts) if p == "xorq_gallery")
    return ".".join(parts[idx:]).removesuffix(".py")


def get_exprs_dict(on_script=None):
    result = {}
    for script in scripts:
        if on_script:
            on_script(script.name)
        exprs = cexcepts(Exception, get_exprs_for_script)(script)
        if exprs:
            result[script.name] = tuple(exprs)
    return result


def update_exprs_json_cache(on_script=None):
    dct = get_exprs_dict(on_script=on_script)
    for script_name, expr_names in dct.items():
        (exprs_dir / f"{script_name}.json").write_text(
            json.dumps(expr_names, sort_keys=True)
        )
    return exprs_dir


def load_exprs_json_cache():
    return {
        p.name.removesuffix(".json"): json.loads(p.read_text())
        for p in sorted(exprs_dir.glob("*.json"))
    }


@cexcepts(Exception, handler=lambda _: None)
def _build_expr(script, expr_name):
    import xorq.api as xo

    mod = importlib.import_module(_module_name_for_script(script))
    expr = mod.__dict__[expr_name]
    return str(xo.build_expr(expr))


def _build_script_exprs(script_name, script_path, expr_names):
    """Build all exprs for a single script. Runs in a worker process."""
    script = Path(script_path)
    return {
        expr_name: built
        for expr_name in expr_names
        if (built := _build_expr(script, expr_name)) is not None
    }


def _get_build_paths_dict_parallel(items, script_by_name, max_workers, on_script):
    result = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_script = {
            pool.submit(
                _build_script_exprs,
                script_name,
                str(script_by_name[script_name]),
                tuple(expr_names),
            ): script_name
            for script_name, expr_names in items
        }
        for future in as_completed(future_to_script):
            script_name = future_to_script[future]
            if on_script:
                on_script(script_name)
            try:
                result[script_name] = future.result()
            except Exception:
                result[script_name] = {}
    return result


def get_build_paths_dict(script_names=None, on_script=None, max_workers=None):
    script_by_name = {s.name: s for s in scripts}
    cache = load_exprs_json_cache()
    items = [
        (sn, ens)
        for sn, ens in cache.items()
        if sn in script_by_name and (script_names is None or sn in script_names)
    ]
    match max_workers:
        case None | 1:
            result = {}
            for script_name, expr_names in items:
                if on_script:
                    on_script(script_name)
                result[script_name] = {
                    expr_name: built
                    for expr_name in expr_names
                    if (built := _build_expr(script_by_name[script_name], expr_name))
                    is not None
                }
            return result
        case _:
            return _get_build_paths_dict_parallel(
                items, script_by_name, max_workers, on_script
            )


def update_build_paths_json_cache(script_names=None, on_script=None, max_workers=None):
    rebuilt = get_build_paths_dict(
        script_names=script_names, on_script=on_script, max_workers=max_workers
    )
    for script_name, exprs in rebuilt.items():
        (build_paths_dir / f"{script_name}.json").write_text(
            json.dumps(exprs, sort_keys=True)
        )
    return build_paths_dir


def load_build_paths_json_cache():
    return {
        p.name.removesuffix(".json"): json.loads(p.read_text())
        for p in sorted(build_paths_dir.glob("*.json"))
    }


def _make_alias(script_name, expr_name):
    return f"{script_name.removesuffix('.py')}-{expr_name}"


def _desired_catalog_state(build_cache):
    alias_to_entry = tuple(
        (_make_alias(sn, en), bp.split("/")[-1])
        for sn, exprs in build_cache.items()
        for en, bp in exprs.items()
    )
    entries = frozenset(entry for _, entry in alias_to_entry)
    return entries, alias_to_entry


def _get_catalog():
    from git import Repo
    from xorq.catalog.catalog import Catalog

    repo = Repo(Path.cwd(), search_parent_directories=True)
    repo_root = Path(repo.working_dir)
    submodule_rel = Catalog.submodule_rel_path.rstrip("/")
    catalog_sms = [
        sm for sm in repo.submodules if sm.path.startswith(submodule_rel + "/")
    ]
    if len(catalog_sms) == 0:
        raise RuntimeError(
            f"No submodule found under {submodule_rel!r}. "
            f"Registered submodules: {[sm.path for sm in repo.submodules]}. "
            f"Run: bash dev/init-catalog-submodule.sh --empty"
        )
    if len(catalog_sms) > 1:
        raise RuntimeError(
            f"Multiple submodules under {submodule_rel!r}: "
            f"{[sm.path for sm in catalog_sms]}. "
            f"Remove extras with: bash dev/rm-submodule.sh <path>"
        )
    catalog_path = repo_root / catalog_sms[0].path
    return Catalog.from_repo_path(catalog_path, init=False, check_consistency=False)


def _current_catalog_state(catalog):
    entries = frozenset(catalog.list())
    alias_to_entry = tuple(
        (ca.alias, ca.catalog_entry.name) for ca in catalog.catalog_aliases
    )
    return entries, alias_to_entry


@frozen
class CatalogDiff:
    aliases_to_remove: tuple[str, ...] = ()
    entries_to_remove: tuple[str, ...] = ()
    entries_to_add: tuple[tuple[str, tuple[str, ...]], ...] = ()
    aliases_to_add: tuple[tuple[str, str], ...] = ()

    @property
    def is_empty(self):
        return not any(
            (
                self.aliases_to_remove,
                self.entries_to_remove,
                self.entries_to_add,
                self.aliases_to_add,
            )
        )


def compute_catalog_diff(catalog, build_cache):
    desired_entries, desired_alias_to_entry = _desired_catalog_state(build_cache)
    current_entries, current_alias_to_entry = _current_catalog_state(catalog)

    desired_alias_map = dict(desired_alias_to_entry)
    current_alias_map = dict(current_alias_to_entry)

    # Aliases to remove: stale or pointing to wrong entry
    aliases_to_remove = tuple(
        alias
        for alias, entry in current_alias_map.items()
        if desired_alias_map.get(alias) != entry
    )

    # Entries to remove: no longer desired
    entries_to_remove = tuple(sorted(current_entries - desired_entries))

    # Entries to add: new, grouped with their aliases
    new_entry_hashes = desired_entries - current_entries
    entry_to_aliases = {}
    for alias, entry in desired_alias_to_entry:
        entry_to_aliases.setdefault(entry, []).append(alias)
    entries_to_add = tuple(
        (entry_hash, tuple(entry_to_aliases.get(entry_hash, ())))
        for entry_hash in sorted(new_entry_hashes)
    )

    # Aliases to add for already-existing entries (after removals/additions)
    surviving_entries = (
        current_entries - frozenset(entries_to_remove)
    ) | new_entry_hashes
    aliases_after_add = frozenset(
        alias for entry_hash, aliases in entries_to_add for alias in aliases
    )
    aliases_to_add = tuple(
        (alias, entry)
        for alias, entry in desired_alias_to_entry
        if alias not in current_alias_map
        and alias not in aliases_after_add
        and entry in surviving_entries
    )

    return CatalogDiff(
        aliases_to_remove=aliases_to_remove,
        entries_to_remove=entries_to_remove,
        entries_to_add=entries_to_add,
        aliases_to_add=aliases_to_add,
    )


def apply_catalog_diff(catalog, diff, builds_dir, on_step=None):
    from xorq.catalog.catalog import CatalogAlias

    for alias in diff.aliases_to_remove:
        if on_step:
            on_step(f"remove alias {alias}")
        CatalogAlias.from_name(alias, catalog).remove()

    for entry_name in diff.entries_to_remove:
        if on_step:
            on_step(f"remove {entry_name}")
        catalog.remove(entry_name, sync=False)

    for entry_hash, aliases in diff.entries_to_add:
        if on_step:
            on_step(f"add {entry_hash[:12]}")
        build_dir = builds_dir / entry_hash
        assert build_dir.is_dir(), f"Build directory not found: {build_dir}"
        catalog.add(build_dir, sync=False, aliases=aliases)

    for alias, entry_hash in diff.aliases_to_add:
        if on_step:
            on_step(f"alias {alias}")
        catalog.add_alias(entry_hash, alias, sync=False)

    catalog.assert_consistency()
    return catalog


def update_catalog(dry_run=False):
    from git import Repo

    catalog = _get_catalog()
    repo_root = Path(Repo(Path.cwd(), search_parent_directories=True).working_dir)
    builds_dir = repo_root / "builds"
    build_cache = load_build_paths_json_cache()

    diff = compute_catalog_diff(catalog, build_cache)

    if not dry_run:
        apply_catalog_diff(catalog, diff, builds_dir)

    return diff
