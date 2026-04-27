#!/usr/bin/env bash
set -euo pipefail

# Rebuild the catalog submodule using a git-annex remote with an S3-backed
# special remote. Defaults to starting from scratch (--empty). Pass --no-empty
# to clone the existing remote with its history instead.
#
# Usage:
#   ./dev/rebuild-catalog-annex.sh [--no-empty] [--env-file FILE] [--gcs] [--annex-uuid UUID]
#
# --no-empty:   clone the existing remote instead of creating a fresh catalog
# --env-file:   env file with S3 credentials (default: .envrcs/.env.catalog.s3.write)
#               only used with --empty (the default)
# --gcs:        apply GCS defaults to the S3 remote (only with --empty)
# --annex-uuid: use this UUID for git-annex init (default: auto-generated)

REMOTE_URL="git@github.com:xorq-labs/xorq-gallery-sklearn-annex.git"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

source "$(dirname "$0")/_catalog-path.sh"
CATALOG_REL="$(catalog_path_from_gitmodules)"

# --- Parse args ---
empty=true
env_file=".envrcs/.env.catalog.s3.write"
gcs_flag=()
annex_uuid_flag=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) echo "Usage: $0 [--no-empty] [--env-file FILE] [--gcs] [--annex-uuid UUID]"; exit 0 ;;
        --no-empty) empty=false; shift ;;
        --env-file) env_file="$2"; shift 2 ;;
        --gcs) gcs_flag=(--gcs); shift ;;
        --annex-uuid) annex_uuid_flag=(--annex-uuid "$2"); shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "=== Rebuilding git-annex catalog ==="
echo "Remote:   ${REMOTE_URL}"
echo "Path:     ${CATALOG_REL}"
echo "Empty:    ${empty}"
if [[ "$empty" == true ]]; then
    echo "Env file: ${env_file}"
fi
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
echo ""
if [[ "$empty" == true ]]; then
    echo "--- Initializing submodule with git-annex ---"
    bash dev/init-catalog-submodule.sh \
        --remote "$REMOTE_URL" \
        --empty \
        --env-file "$env_file" \
        "${gcs_flag[@]}" \
        "${annex_uuid_flag[@]}"
else
    echo "--- Cloning submodule from remote ---"
    bash dev/init-catalog-submodule.sh --remote "$REMOTE_URL"
fi

echo ""
echo "=== Done ==="
git submodule status "$CATALOG_REL"
git -C "$CATALOG_REL" remote -v
if [[ -d "$(git -C "$CATALOG_REL" rev-parse --absolute-git-dir)/annex" ]]; then
    git -C "$CATALOG_REL" annex info
fi
