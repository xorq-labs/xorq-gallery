#!/usr/bin/env bash
set -euo pipefail

# Add a catalog submodule backed by a GitHub remote.
#
# Usage:
#   ./scripts/init-catalog-submodule.sh [--remote URL] [--empty]
#
# --empty: initialize an empty catalog instead of cloning the remote.
#          Useful for rebuilding the catalog from scratch.
#
# Precondition: no submodule at the catalog path. If one exists,
# remove it first:  bash scripts/rm-submodule.sh .xorq/catalogs/xorq-gallery-sklearn
#
# When switching between branches that use different submodule remotes:
#   git checkout <branch> --recurse-submodules
# or after a regular checkout:
#   git submodule sync && git submodule update --init --force

CATALOG_NAME="xorq-gallery-sklearn"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

# --- Verify submodule path matches xorq's expected path ---
XORQ_SUBMODULE_REL="$(uv run python -c 'from xorq.catalog.catalog import Catalog; print(Catalog.submodule_rel_path)')"
CATALOG_REL="${XORQ_SUBMODULE_REL}/${CATALOG_NAME}"

# --- Fail fast if submodule or leftover directory exists ---
if git config --file .gitmodules --get "submodule.${CATALOG_REL}.path" &>/dev/null 2>&1; then
    echo "Submodule already registered in .gitmodules at ${CATALOG_REL}." >&2
    echo "Remove it first:  bash scripts/rm-submodule.sh ${CATALOG_REL}" >&2
    exit 1
fi
if [[ -d "$CATALOG_REL" ]]; then
    echo "Leftover directory exists at ${CATALOG_REL}." >&2
    echo "Remove it first:  rm -rf ${CATALOG_REL}" >&2
    exit 1
fi
git_dir="$(git rev-parse --absolute-git-dir)"
module_cache="${git_dir}/modules/${CATALOG_REL}"
if [[ -d "$module_cache" ]]; then
    echo "Stale module cache at ${module_cache}." >&2
    echo "Remove it first:  chmod -R u+w '${module_cache}' && rm -rf '${module_cache}'" >&2
    exit 1
fi

# --- Parse args ---
remote_url=""
empty=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote) remote_url="$2"; shift 2 ;;
        --empty) empty=true; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# --- Resolve remote URL ---
if [[ -z "$remote_url" ]]; then
    parent_origin="$(git remote get-url origin)"
    if [[ "$parent_origin" == git@github.com:* ]]; then
        org="$(echo "$parent_origin" | sed 's|git@github.com:||;s|/.*||')"
        remote_url="git@github.com:${org}/${CATALOG_NAME}.git"
    elif [[ "$parent_origin" == https://github.com/* ]]; then
        org="$(echo "$parent_origin" | sed 's|https://github.com/||;s|/.*||')"
        remote_url="https://github.com/${org}/${CATALOG_NAME}.git"
    else
        echo "Cannot derive org from origin: $parent_origin" >&2
        echo "Pass --remote URL explicitly." >&2
        exit 1
    fi
fi

echo "Catalog submodule: ${CATALOG_REL}"
echo "Remote URL:        ${remote_url}"

# --- Ensure the remote repo exists on GitHub ---
gh_repo=""
if [[ "$remote_url" == git@github.com:* ]]; then
    gh_repo="$(echo "$remote_url" | sed 's|git@github.com:||;s|\.git$||')"
elif [[ "$remote_url" == https://github.com/* ]]; then
    gh_repo="$(echo "$remote_url" | sed 's|https://github.com/||;s|\.git$||')"
fi

if [[ -n "$gh_repo" ]]; then
    if gh repo view "$gh_repo" &>/dev/null; then
        echo "Remote repo ${gh_repo} already exists."
    else
        echo "Creating private repo ${gh_repo}..."
        gh repo create "$gh_repo" --private --confirm
    fi
fi

# --- Add submodule ---
if [[ "$empty" == true ]]; then
    # Let xorq create the catalog (it handles git init + structure)
    uv run python -c "
from xorq.catalog.catalog import Catalog
Catalog.from_repo_path('${CATALOG_REL}', init=True)
"
    git -C "$CATALOG_REL" remote add origin "$remote_url"
    git submodule add "$remote_url" "$CATALOG_REL"
else
    git submodule add "$remote_url" "$CATALOG_REL"
    git submodule update --init "$CATALOG_REL"
fi

echo ""
echo "Done. Verify with:"
echo "  git submodule status"
echo "  uv run xorq-gallery update-catalog --dry-run"
