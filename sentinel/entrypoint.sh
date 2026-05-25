#!/usr/bin/env bash
set -e

# Parse REDIS_URL (redis://user:pass@host:port) into SENTINEL_ vars
if [ -n "$REDIS_URL" ]; then
    rest="${REDIS_URL#redis://}"
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

exec sentinel start --config sentinel-cloud.json --mode agent
