# dev/

Developer workflow scripts for managing the catalog submodule.

## Scripts

### `init-catalog-submodule.sh [--remote URL] [--empty]`

Add the catalog submodule. Derives the remote URL from the parent repo's
origin unless `--remote` is given. Creates the GitHub repo if it doesn't
exist. Use `--empty` to initialize a blank catalog instead of cloning.

Precondition: no submodule at the catalog path. Remove first with
`rm-submodule.sh` if one exists.

### `rm-submodule.sh <submodule-path>`

Cleanly remove a submodule: deinit, `git rm`, and delete the module cache.
Requires a commit afterward to finalize.

### `switch-catalog-branch.sh [<branch>]`

Switch to a branch that may use a different submodule remote. Handles URL
syncing, module cache cleanup, and annex symlink repair.

Without arguments, repairs the submodule on the current branch (useful when
a plain `git checkout` left the submodule in a broken state).
