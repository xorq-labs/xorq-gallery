import pathlib


def get_scripts_for_group(group_name):
    group = next((g for g in groups if g.name == group_name), None)
    match group:
        case None:
            raise ValueError(f"no such group {group_name!r}")
        case _:
            return tuple(p for p in sorted(group.glob("*.py")) if p.name != "__init__.py")


groups = tuple(
    p
    for p in sorted(pathlib.Path(__file__).parent.iterdir())
    if p.is_dir() and not p.name.startswith("_")
)
scripts = tuple(
    p
    for group in groups
    for p in get_scripts_for_group(group.name)
)
