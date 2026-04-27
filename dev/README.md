# dev/

Developer workflow scripts for managing the catalog submodule.

## Catalog name convention

The catalog name (`xorq-gallery-sklearn`) is hardcoded in exactly one place:
`init-catalog-submodule.sh`, because the submodule does not exist in
`.gitmodules` yet when that script runs. All other scripts and Python code
derive the catalog path at runtime from `.gitmodules` (shell) or
`repo.submodules` (Python), so renaming the catalog only requires updating
the init script.

## Shared helpers

- `_catalog-path.sh` — `catalog_path_from_gitmodules`: derives the single catalog
  submodule path from `.gitmodules`. Returns empty string if none, exits 1 if multiple.
- `_validate-env.sh` — `validate_env_file`: rejects env files with shell metacharacters.

## Scripts

### `init-catalog-submodule.sh [--remote URL] [--empty] [--env-file FILE] [--gcs]`

Add the catalog submodule. Derives the remote URL from the parent repo's
origin unless `--remote` is given. Creates the GitHub repo if it doesn't
exist. Use `--empty` to initialize a blank catalog instead of cloning.

With `--env-file`, also initializes git-annex and creates an S3 special
remote using credentials and config from the env file (requires `--empty`).
Add `--gcs` to apply Google Cloud Storage defaults (host, protocol, etc.).

Precondition: no submodule at the catalog path. Remove first with
`reset-catalog.sh` or `rm-submodule.sh`.

### `reset-catalog.sh [--env-file FILE] [--dry-run] [--force] [--submodule-only]`

Tear down the catalog: remove the submodule and clear its S3 bucket prefix.
Uses write credentials from the env file (default: `.envrcs/.env.catalog.s3.write`).
Prompts for confirmation before deleting. Use `--dry-run` to preview without acting.
Use `--force` to skip the confirmation prompt (for CI).
Use `--submodule-only` to remove just the submodule without touching S3.
Leaves staged deletions from `git rm`; run `git commit` afterward to finalize.

### `rm-submodule.sh [--force] <submodule-path>`

Cleanly remove a submodule: deinit, `git rm`, and delete the module cache.
With `--force`, also handles leftover directories and stale caches when the
submodule is no longer registered in `.gitmodules`.
Requires a commit afterward to finalize.

### `rebuild-catalog-git.sh [--no-empty]`

Tear down and reinitialize the catalog submodule using the plain git remote
(`xorq-gallery-sklearn.git`). Defaults to creating a fresh, empty catalog.
Pass `--no-empty` to clone the existing remote with its full history instead.

Composes `rm-submodule.sh` and `init-catalog-submodule.sh`.

### `rebuild-catalog-annex.sh [--no-empty] [--env-file FILE] [--gcs]`

Tear down and reinitialize the catalog submodule using the git-annex remote
(`xorq-gallery-sklearn-annex.git`). Defaults to creating a fresh, empty
catalog with git-annex initialized via S3 credentials from the env file
(default: `.envrcs/.env.catalog.s3.write`). Add `--gcs` for Google Cloud
Storage defaults. Pass `--no-empty` to clone the existing remote instead
(skips annex init and env file).

Composes `rm-submodule.sh` and `init-catalog-submodule.sh`.

### Populating the catalog after rebuild

The rebuild scripts only set up the empty submodule. To populate it, run
the update chain in order — each step reads from the previous:

```bash
xorq-gallery update-exprs          # scan scripts → data/exprs/{script}.json
xorq-gallery update-build-paths    # build each expr → data/build_paths/{script}.json
xorq-gallery update-catalog        # diff build_paths vs catalog → add/remove entries and aliases
```

### Updating the git-annex catalog (`feat/catalog/git-annex`)

Full rebuild and populate cycle for the annex-backed catalog:

```bash
# 1. Rebuild the empty annex submodule with S3 write credentials
rebuild-catalog-annex.sh --env-file ./.envrcs/.env.catalog.s3.write --gcs

# 2. Replay snapshots into the catalog
xorq catalog --path ../xorq-gallery-sklearn replay \
  --env-file ./.envrcs/.env.catalog.s3.write --gcs \
  ./.xorq/catalogs/xorq-gallery-sklearn

# 3. Embed read-only credentials for consumers
xorq catalog embed-readonly --env-file ./.envrcs/.env.catalog.s3.read --gcs
```

### `switch-catalog-branch.sh [<branch>]`

Switch to a branch that may use a different submodule remote. Plain
`git checkout` does not update the cached submodule URL, so a subsequent
`git submodule update` tries to fetch commits from the wrong remote.

This script:
1. Checks out the branch (skipped if already on it)
2. Clears the module cache and working tree when the URL changes
   (prevents stale annex metadata from leaking across remotes)
3. Runs `git submodule sync && update --init --force`
4. Runs `git annex fix` if the submodule is annex-backed

Without arguments, repairs the submodule on the current branch (useful when
a plain `git checkout` left the submodule in a broken state).
