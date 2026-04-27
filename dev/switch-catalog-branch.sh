#!/usr/bin/env bash
set -euo pipefail

# Switch to a branch that may use a different submodule remote for the catalog.
#
# Usage:
#   ./dev/switch-catalog-branch.sh [<branch>]
#
# If <branch> is omitted, repairs the submodule on the current branch
# (syncs URL, re-inits, fixes annex symlinks).
#
# This handles what `git checkout --recurse-submodules` does not:
# 1. Checkout the branch (if given)
# 2. Sync the submodule URL from .gitmodules
# 3. Force-update the submodule to match the new remote
# 4. If the submodule is annex-backed, fix symlinks

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [<branch>]"
    exit 0
fi

branch="${1:-$(git branch --show-current)}"
repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

source "$(dirname "$0")/_catalog-path.sh"
CATALOG_REL="$(catalog_path_from_gitmodules)"
if [[ -z "$CATALOG_REL" ]]; then
    echo "No submodule found in .gitmodules" >&2
    exit 1
fi

# --- Show what's changing ---
current_url="$(git config --file .gitmodules --get "submodule.${CATALOG_REL}.url" 2>/dev/null || echo "(none)")"
target_url="$(git show "${branch}:.gitmodules" 2>/dev/null | git config --file /dev/stdin --get "submodule.${CATALOG_REL}.url" 2>/dev/null || echo "(none)")"

echo "Branch:     $(git branch --show-current) -> ${branch}"
echo "Submodule:  ${CATALOG_REL}"
echo "Remote URL: ${current_url} -> ${target_url}"

# --- Checkout (skip if already on target) ---
current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$branch" ]]; then
    git checkout "$branch"
fi

# --- Clean module cache when the remote changes ---
# Prevents stale annex metadata from leaking into plain-git checkouts (and vice versa).
git_dir="$(git rev-parse --absolute-git-dir)"
module_cache="${git_dir}/modules/${CATALOG_REL}"
if [[ "$current_url" != "$target_url" && -d "$module_cache" ]]; then
    echo "Remote changed, clearing module cache and working tree..."
    chmod -R u+w "$module_cache"
    rm -rf "$module_cache"
    rm -rf "$CATALOG_REL"
fi

# --- Sync URL and force-update submodule ---
git submodule sync
git submodule update --init --force

# --- Fix annex symlinks if needed ---
if [[ -d "${CATALOG_REL}/.git" ]] || [[ -f "${CATALOG_REL}/.git" ]]; then
    git_dir_resolved="$(git -C "$CATALOG_REL" rev-parse --absolute-git-dir)"
    if [[ -d "${git_dir_resolved}/annex" ]]; then
        echo "Annex-backed catalog detected, fixing symlinks..."
        git -C "$CATALOG_REL" annex fix 2>&1 | grep -v "Unable to parse git config" | grep -v "annex-ignore" | grep -v "git-annex-shell" || true
    fi
fi

echo ""
echo "Done. Current submodule state:"
git submodule status "$CATALOG_REL"
