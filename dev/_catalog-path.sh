#!/usr/bin/env bash
# Derive the single catalog submodule path from .gitmodules.
# Returns "" if no submodule is registered; exits 1 if multiple are found.
#
# Usage: source dev/_catalog-path.sh && catalog_path_from_gitmodules

catalog_path_from_gitmodules() {
    local paths
    paths="$(git config --file .gitmodules --get-regexp 'submodule\..*\.path' 2>/dev/null | awk '{print $2}')"
    if [[ -z "$paths" ]]; then
        echo ""
        return 0
    elif [[ "$(echo "$paths" | wc -l)" -gt 1 ]]; then
        echo "Multiple submodules in .gitmodules; cannot auto-detect catalog:" >&2
        echo "$paths" >&2
        return 1
    fi
    echo "$paths"
}
