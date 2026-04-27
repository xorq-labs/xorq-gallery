#!/usr/bin/env bash
set -euo pipefail

# Remove the catalog submodule and clear its S3 bucket prefix.
#
# Usage:
#   ./dev/reset-catalog.sh [--env-file FILE] [--dry-run] [--force] [--submodule-only] [--annex-uuid UUID]
#
# --env-file:       env file with write credentials (default: .envrcs/.env.catalog.s3.write)
# --dry-run:        print what would be deleted without acting
# --force:          skip confirmation prompt (for CI)
# --submodule-only: remove the submodule without touching S3
# --annex-uuid:     annex UUID to derive S3 prefix from (default: read from submodule)
#
# After reset, re-init with:
#   bash dev/init-catalog-submodule.sh --remote URL --empty --env-file FILE --gcs

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

source "$(dirname "$0")/_catalog-path.sh"
CATALOG_REL="$(catalog_path_from_gitmodules)" || exit 1

# --- Parse args ---
env_file=".envrcs/.env.catalog.s3.write"
dry_run=false
force=false
submodule_only=false
annex_uuid=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) echo "Usage: $0 [--env-file FILE] [--dry-run] [--force] [--submodule-only] [--annex-uuid UUID]"; exit 0 ;;
        --env-file) env_file="$2"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --force) force=true; shift ;;
        --submodule-only) submodule_only=true; shift ;;
        --annex-uuid) annex_uuid="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

bucket=""
prefix=""
endpoint_args=()
if [[ "$submodule_only" != true ]]; then
    source "$(dirname "$0")/_validate-env.sh"
    validate_env_file "$env_file"

    # --- Load env for bucket info ---
    set -a; source "$env_file"; set +a
    bucket="${XORQ_CATALOG_S3_BUCKET:?missing XORQ_CATALOG_S3_BUCKET in $env_file}"

    # --- Derive S3 prefix from annex UUID ---
    if [[ -z "$annex_uuid" ]]; then
        if [[ -n "$CATALOG_REL" ]] && [[ -d "$CATALOG_REL" ]]; then
            annex_uuid="$(git -C "$CATALOG_REL" config annex.uuid 2>/dev/null || true)"
        fi
        if [[ -z "$annex_uuid" ]]; then
            echo "Cannot determine annex UUID from submodule." >&2
            echo "Pass --annex-uuid UUID explicitly." >&2
            exit 1
        fi
    fi
    prefix="annex-only/${annex_uuid}/"

    # --- Build endpoint URL from env vars ---
    host="${XORQ_CATALOG_S3_HOST:-}"
    protocol="${XORQ_CATALOG_S3_PROTOCOL:-https}"
    if [[ -n "$host" ]]; then
        endpoint_args+=(--endpoint-url "${protocol}://${host}")
    fi
fi

if [[ "$dry_run" == true ]]; then
    if [[ "$submodule_only" != true ]]; then
        echo "[dry-run] Would delete s3://${bucket}/${prefix} (recursive)"
        if [[ ${#endpoint_args[@]} -gt 0 ]]; then
            echo "[dry-run] Endpoint: ${endpoint_args[*]}"
        fi
    fi
    if [[ -n "$CATALOG_REL" ]]; then
        echo "[dry-run] Would remove submodule ${CATALOG_REL}"
    else
        echo "[dry-run] No submodule in .gitmodules, skipping submodule removal"
    fi
    exit 0
fi

# --- Confirm ---
if [[ "$force" != true ]]; then
    echo "This will:"
    if [[ "$submodule_only" != true ]]; then
        echo "  - Delete all objects under s3://${bucket}/${prefix}"
    fi
    if [[ -n "$CATALOG_REL" ]]; then
        echo "  - Remove submodule ${CATALOG_REL}"
    else
        echo "  - (no submodule in .gitmodules to remove)"
    fi
    read -p "Proceed? [y/N] " confirm
    if [[ "$confirm" != [yY] ]]; then
        echo "Aborted." >&2
        exit 1
    fi
fi

# --- Clear bucket prefix ---
# S3 deletion must succeed before we tear down the submodule.  Use
# --submodule-only to skip this step.
if [[ "$submodule_only" != true ]]; then
    echo "Clearing s3://${bucket}/${prefix}..."
    AWS_ACCESS_KEY_ID="${XORQ_CATALOG_S3_AWS_ACCESS_KEY_ID:?missing}" \
    AWS_SECRET_ACCESS_KEY="${XORQ_CATALOG_S3_AWS_SECRET_ACCESS_KEY:?missing}" \
    aws s3 rm "${endpoint_args[@]}" \
        "s3://${bucket}/${prefix}" --recursive
fi

# --- Remove submodule ---
if [[ -n "$CATALOG_REL" ]]; then
    echo "Removing submodule ${CATALOG_REL}..."
    bash dev/rm-submodule.sh --force "$CATALOG_REL"
else
    echo "No submodule in .gitmodules, skipping submodule removal."
fi

echo ""
if [[ "$submodule_only" == true ]]; then
    echo "Done. Submodule removed."
elif [[ -n "$CATALOG_REL" ]]; then
    echo "Done. Submodule removed, bucket prefix cleared."
else
    echo "Done. Bucket prefix cleared."
fi
if ! git diff --cached --quiet 2>/dev/null; then
    echo "Finalize with:"
    echo "  git commit -m 'chore: reset catalog'"
fi
