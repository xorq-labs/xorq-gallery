"""Run `xorq build` on every expr listed in the exprs.json cache.

Usage (from repo root):
    python scripts/build_all_exprs_from_cache.py
"""

from __future__ import annotations

import sys

from xorq_gallery.sklearn.utils import (
    load_build_paths_json_cache,
    load_exprs_json_cache,
    update_build_paths_json_cache,
)


def main():
    exprs_cache = load_exprs_json_cache()
    total_exprs = sum(len(v) for v in exprs_cache.values())

    print(f"Building {total_exprs} exprs from {len(exprs_cache)} scripts...")
    path = update_build_paths_json_cache()

    build_cache = load_build_paths_json_cache()
    built = sum(len(v) for v in build_cache.values())
    failed = total_exprs - built

    for script_name, expr_paths in sorted(build_cache.items()):
        print(f"\n=== {script_name} ===")
        for expr_name, build_path in expr_paths.items():
            print(f"  OK:   {expr_name}  ->  {build_path}")

    # report any that didn't make it
    if failed:
        print(f"\nFailed ({failed}):")
        for script_name, expr_names in sorted(exprs_cache.items()):
            built_names = set(build_cache.get(script_name, {}))
            for expr_name in expr_names:
                if expr_name not in built_names:
                    print(f"  {script_name}:{expr_name}")

    print(f"\n{'=' * 60}")
    print(f"Results: {built} passed, {failed} failed")
    print(f"Cache written to {path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
