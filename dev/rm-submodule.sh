#!/usr/bin/env bash
set -euo pipefail

force=false
submodule_path=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) force=true; shift ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            if [[ -n "$submodule_path" ]]; then
                echo "Usage: $0 [--force] <submodule-path>" >&2
                exit 1
            fi
            submodule_path="$1"; shift ;;
    esac
done

if [[ -z "$submodule_path" ]]; then
    echo "Usage: $0 [--force] <submodule-path>" >&2
    exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
git_dir="$(git rev-parse --absolute-git-dir)"

# Normalize to a repo-root-relative path
abs_path="$(cd "$repo_root" && realpath --relative-to=. "$(realpath -m "$submodule_path")")"
submodule_path="${abs_path%/}"

if git config --file "$repo_root/.gitmodules" --get "submodule.${submodule_path}.path" &>/dev/null; then
    cd "$repo_root"
    git submodule deinit -f "$submodule_path"
    git rm -f "$submodule_path"
elif [[ "$force" == true ]]; then
    if [[ -d "$repo_root/$submodule_path" ]]; then
        echo "Removing leftover directory ${submodule_path}..."
        rm -rf "$repo_root/$submodule_path"
    else
        echo "No directory at ${submodule_path}, nothing to remove."
    fi
else
    echo "Error: '${submodule_path}' is not a registered submodule (use --force to remove leftovers)" >&2
    exit 1
fi

module_cache="${git_dir}/modules/${submodule_path}"
if [ -d "$module_cache" ]; then
    chmod -R u+w "$module_cache"
    rm -rf "$module_cache"
    echo "Removed module cache: ${module_cache}"
fi

echo "Submodule '${submodule_path}' removed. Commit to finalize."
