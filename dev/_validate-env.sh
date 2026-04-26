#!/usr/bin/env bash
# Validate that an env file contains only safe KEY=VALUE lines.
# Rejects shell metacharacters ($, `, ;, |, &) in values to prevent
# command injection when the file is later sourced.
#
# Values must be unquoted: KEY=value, not KEY="value" or KEY='value'.
#
# Usage: source dev/_validate-env.sh && validate_env_file "$path"

validate_env_file() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo "Env file not found: $file" >&2
        return 1
    fi
    # Allow: blank lines, comments, KEY=value, export KEY=value.
    # Reject values containing shell metacharacters: $ ` ; | & ( )
    if grep -qvE '^\s*(#.*|((export\s+)?[A-Za-z_][A-Za-z_0-9]*=[^$`;|&()]*))?\s*$' "$file"; then
        echo "Env file contains invalid lines (expected KEY=VALUE without shell metacharacters): $file" >&2
        grep -nvE '^\s*(#.*|((export\s+)?[A-Za-z_][A-Za-z_0-9]*=[^$`;|&()]*))?\s*$' "$file" >&2
        return 1
    fi
}
