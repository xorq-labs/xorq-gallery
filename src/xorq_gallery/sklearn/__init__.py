import pathlib


def get_scripts_for_group(group):
    group = next((g for g in group_paths if g.name == group), None)
    match group:
        case None:
            raise ValueError(f"no such group {group!r}")
        case _:
            return tuple(
                p for p in sorted(group.glob("*.py")) if p.name != "__init__.py"
            )


group_paths = tuple(
    p
    for p in sorted(pathlib.Path(__file__).parent.iterdir())
    if p.is_dir() and not p.name.startswith("_") and not p.name.startswith(".")
)
scripts = tuple(
    p for group_path in group_paths for p in get_scripts_for_group(group_path.name)
)
group_to_scripts = tuple(
    (group.name, get_scripts_for_group(group.name)) for group in group_paths
)
