# ADR-001: Separate caches for exprs and build paths

## Status

Accepted

## Context

Every gallery script exposes deferred xorq expressions, and each expression produces a build artifact via `xo.build_expr`. We need to cache two mappings:

1. **script name -> expr names** (which expressions does each script export?)
2. **(script, expr) -> build hash** (what artifact did each expression produce?)

The simplest option is a single JSON file combining both. The alternative is two independent files with their own update/load functions.

## Decision

Use two separate cache files: `exprs.json` and `build_paths.json`, each with symmetric `update_*` / `load_*` functions.

## Rationale

- **Different lifecycles.** The exprs cache is cheap to regenerate (import scripts, inspect `__dict__`). The build cache is expensive (~7 minutes to rebuild all expressions). Coupling them means the cheap operation can't run without the expensive one, or vice versa.
- **Different consumers.** Code that enumerates expressions (CLI `list-exprs`, test parametrization) doesn't need build paths. Code that syncs the catalog doesn't need to re-derive expr names.
- **Producer/consumer relationship.** The build cache iterates over the exprs cache to know what to build. This dependency is natural when they're separate; it would become a circular concern inside a single file.
- **Independent staleness.** Adding a new script invalidates only the exprs cache. Changing an expression's internals invalidates only the build cache. Separate files let each be refreshed independently.

## Consequences

- Two files to keep in sync rather than one. Mitigated by the validation chain (ADR-004) which checks that build_paths.json covers every entry in exprs.json.
- Callers that need both mappings make two `load_*` calls. In practice this is only `update_catalog` and tests.
