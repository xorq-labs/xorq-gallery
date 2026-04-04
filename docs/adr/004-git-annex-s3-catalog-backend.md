# ADR-004: git-annex S3 backend for the catalog submodule

## Status

Accepted

## Context

The catalog submodule (`.xorq/catalogs/xorq-gallery-sklearn`) stores build artifacts as `.zip` entries committed directly to git. At ~175 MB across 130 entries, the repo is already large and grows with every new build. Every clone downloads all artifacts, most of which are never accessed. The gallery is meant to demonstrate reproducible deferred pipelines — the catalog should be easy to clone and query without pulling every artifact.

## Decision

Back the catalog with git-annex using an S3-compatible remote (GCS) with embedded read-only credentials (`embedcreds=yes`). The git repo stores only symlinks and metadata; actual `.zip` content lives in S3 and is fetched on demand.

### Key choices

**Embedded credentials.** The S3 remote config (including credentials) is stored in the `git-annex` branch's `remote.log`. This makes cloning zero-config — no environment variables or credential files needed. The embedded credentials are read-only, scoped to a single bucket prefix. Write access uses separate credentials passed via environment at catalog-update time.

**Same submodule path, different remote.** The annex-backed repo (`xorq-gallery-sklearn-annex`) uses the same directory structure as the plain-git repo (`xorq-gallery-sklearn`). The submodule path stays `.xorq/catalogs/xorq-gallery-sklearn` — only the URL in `.gitmodules` changes. This means no code changes in consumers (`utils.py`, test infrastructure, CLI).

**Lazy content fetching.** The `Catalog` class auto-detects the `git-annex` branch, runs `annex init` and `enableremote`, and fetches entry content on demand via `entry.fetch()`. After `git clone --recurse-submodules`, entry files are broken symlinks until individually accessed.

## Consequences

**Adds git-annex as a dependency.** Contributors and CI need `git-annex` installed. Without it, the catalog submodule clones fine but entries are inaccessible broken symlinks.

**Submodule URL switching is fragile.** `git checkout` does not sync the cached submodule URL from `.gitmodules`. Switching between branches that use different catalog remotes (plain-git vs annex) requires `git submodule sync` or the `dev/switch-catalog-branch.sh` helper, which also clears the module cache to prevent stale annex metadata from leaking across remotes.

**Credentials are permanent.** Embedded creds are stored in git-annex branch history. Rotation requires a new remote. This is acceptable for read-only credentials scoped to a single prefix, but write credentials must never be embedded.
