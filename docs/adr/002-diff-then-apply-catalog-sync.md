# ADR-002: Diff-then-apply pattern for catalog sync

## Status

Accepted

## Context

The `xorq-gallery-sklearn` catalog must stay in sync with `build_paths.json`. Syncing involves adding new entries, removing stale ones, and reconciling aliases. A straightforward approach would be a single `update_catalog()` function that reads current state, computes changes, and applies them in one pass.

## Decision

Split catalog sync into three layers:

1. **`compute_catalog_diff(catalog, build_cache) -> CatalogDiff`** -- pure function that computes what needs to change without touching the catalog.
2. **`apply_catalog_diff(catalog, diff, builds_dir)`** -- applies the diff to the catalog.
3. **`update_catalog(dry_run=False) -> CatalogDiff`** -- orchestrator that always returns the diff, only applies when `dry_run=False`.

`CatalogDiff` is a frozen attrs class with four fields: `aliases_to_remove`, `entries_to_remove`, `entries_to_add`, `aliases_to_add`, and an `is_empty` property.

## Rationale

- **Dry-run.** Operators can preview what will change before mutating a git-backed catalog. This is important because each catalog operation creates a git commit in the submodule.
- **Testability.** `compute_catalog_diff` is pure -- it can be tested with synthetic inputs without needing a real catalog on disk.
- **Inspectability.** The diff is a first-class data structure that can be logged, serialized, or used in CI checks. Contrast with an imperative approach where changes are only visible through side effects.
- **Separation of concerns.** The diffing logic doesn't know about filesystem paths or git commits. The application logic doesn't know about desired-vs-current state comparison.

## Consequences

- More functions to maintain than a monolithic sync. Acceptable given the three are small and the boundaries are natural.
- `CatalogDiff` is immutable, so the diff cannot be modified after computation. This is intentional -- if the desired state changes, recompute the diff.
