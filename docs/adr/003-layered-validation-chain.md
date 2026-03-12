# ADR-003: Layered validation chain with fast/slow separation

## Status

Accepted

## Context

Three caches form a chain of truth: `exprs.json` -> `build_paths.json` -> catalog. Each link can go stale independently. We need tests that catch staleness, but rebuilding all expressions to verify hashes takes ~7 minutes.

## Decision

Validate with five tests, each checking one link in the chain:

| Step | Test | What it checks | Speed |
|------|------|----------------|-------|
| 1 | `test_load_exprs_json_cache_matches_get_exprs_dict` | exprs.json == live scripts | fast |
| 2 | `test_load_build_paths_json_cache_keys_match_exprs_cache` | build_paths.json covers every expr | fast |
| 3 | `test_build_paths_json_cache_hashes_are_current` | build hashes == `xo.build_expr` output | slow2 |
| 4 | `test_build_paths_json_cache_dirs_exist` | every hash has a `builds/` directory | fast |
| 5 | `test_catalog_matches_build_paths` | catalog entries/aliases == build_paths.json | fast |

Step 3 is marked `slow2` (>17s). The rest complete in seconds.

## Rationale

- **Granular failure messages.** A single "is everything consistent?" test would fail with an opaque error. Five tests pinpoint exactly which link broke: did someone add a script without updating exprs.json? Did a code change shift a build hash? Did a build directory get deleted?
- **Fast CI, thorough nightly.** The default `pytest -m "not slow2"` runs steps 1, 2, 4, 5 in under a minute. Full validation including step 3 can run on a schedule or before releases.
- **Each test is independently useful.** Step 4 (dirs exist) catches the case where someone cleans `builds/` without updating the cache. Step 5 (catalog matches) catches manual catalog edits that diverge from the cache.
- **Ordered by cost.** Cheaper checks run first. If step 1 fails (exprs.json stale), there's no point running step 3 (rebuild all hashes). Pytest's default ordering respects file order.

## Consequences

- Five tests instead of one, but each is small (5-15 lines) and tests a clear invariant.
- Step 3 is expensive enough that developers may skip it locally. This is acceptable -- CI catches it, and the fast tests catch the most common failures (new scripts, deleted builds, catalog drift).
