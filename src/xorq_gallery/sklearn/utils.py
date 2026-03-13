import importlib
import json
from pathlib import Path

from attr import frozen
from toolz.curried import excepts as cexcepts

from xorq_gallery.sklearn import (
    get_exprs_for_script,
    scripts,
)
from xorq_gallery.utils import get_data_file


exprs_json_path = get_data_file("exprs.json")
build_paths_json_path = get_data_file("build_paths.json")


def _module_name_for_script(script):
    parts = script.parts
    idx = next(i for i, p in enumerate(parts) if p == "xorq_gallery")
    return ".".join(parts[idx:]).removesuffix(".py")


def get_exprs_dict():
    return {
        script.name: tuple(exprs)
        for (script, exprs) in (
            (script, cexcepts(Exception, get_exprs_for_script)(script))
            for script in scripts
        )
        if exprs
    }


def update_exprs_json_cache():
    dct = get_exprs_dict()
    exprs_json_path.write_text(json.dumps(dct))
    return exprs_json_path


def load_exprs_json_cache():
    return json.loads(exprs_json_path.read_text())


@cexcepts(Exception, handler=lambda _: None)
def _build_expr(script, expr_name):
    import xorq.api as xo

    mod = importlib.import_module(_module_name_for_script(script))
    expr = mod.__dict__[expr_name]
    return str(xo.build_expr(expr))


def get_build_paths_dict(script_names=None):
    script_by_name = {s.name: s for s in scripts}
    cache = load_exprs_json_cache()
    return {
        script_name: {
            expr_name: result
            for expr_name in expr_names
            if (result := _build_expr(script_by_name[script_name], expr_name))
            is not None
        }
        for script_name, expr_names in cache.items()
        if script_name in script_by_name
        and (script_names is None or script_name in script_names)
    }


def update_build_paths_json_cache(script_names=None):
    rebuilt = get_build_paths_dict(script_names=script_names)
    if script_names is not None:
        dct = load_build_paths_json_cache()
        dct.update(rebuilt)
    else:
        dct = rebuilt
    build_paths_json_path.write_text(json.dumps(dct))
    return build_paths_json_path


def load_build_paths_json_cache():
    return json.loads(build_paths_json_path.read_text())


CATALOG_NAME = "xorq-gallery-sklearn"


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

    repo_root = Path(Repo(Path.cwd(), search_parent_directories=True).working_dir)
    catalog_path = repo_root / Catalog.submodule_rel_path / CATALOG_NAME
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


def apply_catalog_diff(catalog, diff, builds_dir):
    from xorq.catalog.catalog import CatalogAlias

    for alias in diff.aliases_to_remove:
        CatalogAlias.from_name(alias, catalog).remove()

    for entry_name in diff.entries_to_remove:
        catalog.remove(entry_name, sync=False)

    for entry_hash, aliases in diff.entries_to_add:
        build_dir = builds_dir / entry_hash
        assert build_dir.is_dir(), f"Build directory not found: {build_dir}"
        catalog.add(build_dir, sync=False, aliases=aliases)

    for alias, entry_hash in diff.aliases_to_add:
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
