#!/usr/bin/env bash
set -euo pipefail

# Rebuild the catalog submodule using a plain git remote.
# Defaults to starting from scratch (--empty). Pass --no-empty to clone
# the existing remote with its history instead.
#
# Usage:
#   ./dev/rebuild-catalog-git.sh [--no-empty]

REMOTE_URL="git@github.com:xorq-labs/xorq-gallery-sklearn.git"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

source "$(dirname "$0")/_catalog-path.sh"
CATALOG_REL="$(catalog_path_from_gitmodules)"

# --- Parse args ---
empty=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) echo "Usage: $0 [--no-empty]"; exit 0 ;;
        --no-empty) empty=false; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "=== Rebuilding plain git catalog ==="
echo "Remote: ${REMOTE_URL}"
echo "Path:   ${CATALOG_REL}"
echo "Empty:  ${empty}"
echo ""

# 1. Tear down existing submodule
if [[ -n "$CATALOG_REL" ]] && git config --file .gitmodules --get "submodule.${CATALOG_REL}.path" &>/dev/null; then
    echo "--- Removing existing submodule ---"
    bash dev/rm-submodule.sh --force "$CATALOG_REL"
fi

# Clean up any leftover directory
if [[ -d "$CATALOG_REL" ]]; then
    rm -rf "$CATALOG_REL"
fi

# 2. Re-init
empty_flag=()
if [[ "$empty" == true ]]; then
    empty_flag=(--empty)
fi

echo ""
echo "--- Initializing submodule ---"
bash dev/init-catalog-submodule.sh --remote "$REMOTE_URL" "${empty_flag[@]}"

echo ""
echo "=== Done ==="
git submodule status "$CATALOG_REL"
git -C "$CATALOG_REL" remote -v
