#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <submodule-path>" >&2
    exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
git_dir="$(git rev-parse --absolute-git-dir)"

# Normalize to a repo-root-relative path
abs_path="$(cd "$repo_root" && realpath --relative-to=. "$(realpath -m "$1")")"
submodule_path="${abs_path%/}"

if ! git config --file "$repo_root/.gitmodules" --get "submodule.${submodule_path}.path" &>/dev/null; then
    echo "Error: '${submodule_path}' is not a registered submodule" >&2
    exit 1
fi

cd "$repo_root"
git submodule deinit -f "$submodule_path"
git rm -f "$submodule_path"

module_cache="${git_dir}/modules/${submodule_path}"
if [ -d "$module_cache" ]; then
    rm -rf "$module_cache"
    echo "Removed module cache: ${module_cache}"
fi

echo "Submodule '${submodule_path}' removed. Commit to finalize."
