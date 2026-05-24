#!/usr/bin/env bash
set -e

# Parse REDIS_URL (redis://user:pass@host:port) into SENTINEL_ vars
if [ -n "$REDIS_URL" ]; then
    # Strip scheme
    rest="${REDIS_URL#redis://}"
    # Extract user:pass@host:port
    if [[ "$rest" == *"@"* ]]; then
        userpass="${rest%@*}"
        hostport="${rest##*@}"
        SENTINEL_REDIS_PASSWORD="${userpass##*:}"
        export SENTINEL_REDIS_PASSWORD
    else
        hostport="$rest"
    fi
    SENTINEL_REDIS_HOST="${hostport%:*}"
    SENTINEL_REDIS_PORT="${hostport##*:}"
    export SENTINEL_REDIS_HOST
    export SENTINEL_REDIS_PORT
fi

exec sentinel deploy \
    --config sentinel.json \
    --sentinel-url "$SENTINEL_URL"
